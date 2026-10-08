from __future__ import annotations

import argparse
import keyword
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("ID", "Target", "IndependentVars")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one LLM-SR prompt per Excel row using the ID, Target, "
            "and IndependentVars columns."
        )
    )
    parser.add_argument("excel", type=Path, help="Path to the source .xlsx file")
    parser.add_argument("template", type=Path, help="Path to prompt template .txt")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_prompts",
        help="Output directory (default: generated_prompts beside this script)",
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help="Excel sheet name or zero-based index (default: first sheet)",
    )
    return parser.parse_args()


def require_python_identifier(value: str, field: str, row_id: str) -> str:
    value = value.strip().lstrip("\ufeff")
    if not value or not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError(
            f"{row_id}: {field}={value!r} is not a valid Python identifier"
        )
    return value


def parse_independent_vars(value: object, row_id: str) -> list[str]:
    if pd.isna(value):
        raise ValueError(f"{row_id}: IndependentVars is empty")

    variables = [part.strip() for part in str(value).split(",")]
    if not variables or any(not variable for variable in variables):
        raise ValueError(f"{row_id}: invalid IndependentVars={value!r}")

    variables = [
        require_python_identifier(variable, "IndependentVars", row_id)
        for variable in variables
    ]

    if len(set(variables)) != len(variables):
        raise ValueError(f"{row_id}: IndependentVars contains duplicate names")
    if "params" in variables:
        raise ValueError(f"{row_id}: IndependentVars cannot contain 'params'")
    return variables


def english_join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def replace_once(
    text: str,
    pattern: str,
    replacement: str,
    description: str,
    *,
    flags: int = 0,
) -> str:
    result, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=flags,
    )
    if count != 1:
        raise ValueError(
            f"Template does not match the expected structure: {description}"
        )
    return result


def render_prompt(template: str, target: str, variables: list[str]) -> str:
    """Render one prompt while keeping all non-variable template text unchanged."""
    text = template.replace("\r\n", "\n").replace("\r", "\n")
    variable_csv = ", ".join(variables)
    variable_words = english_join(variables)

    input_lines = "\n".join(
        f"    {variable}: independent variable" for variable in variables
    )
    text = replace_once(
        text,
        r"(?ms)^Input:\n.*?(?=\nOutput:\n)",
        f"Input:\n{input_lines}",
        "Input section",
    )

    text = replace_once(
        text,
        r"(?m)(^Output:\n)    [^\n]+",
        f"Output:\n    {target}: target physical quantity",
        "Output section",
    )

    text = replace_once(
        text,
        r"(?m)^    target = f\([^\n]+\)$",
        f"    {target} = f({variable_csv}, params)",
        "task formula",
    )

    dataset_arguments = "\n".join(
        [*(f"        {variable}," for variable in variables), "        params_k"]
    )
    text = replace_once(
        text,
        r"(?ms)^    target_k = f\(\n.*?^    \)$",
        f"    {target}_k = f(\n{dataset_arguments}\n    )",
        "per-dataset formula",
    )

    text = replace_once(
        text,
        r"The equation may contain physically meaningful nonlinear relationships\n"
        r"between [^\n]+\.",
        "The equation may contain physically meaningful nonlinear relationships\n"
        f"between {variable_words}.",
        "nonlinear-relationship description",
    )

    text = replace_once(
        text,
        r"(?m)^    - use [^\n]+ as independent variables$",
        f"    - use {variable_words} as independent variables",
        "independent-variable requirement",
    )

    assignments = "\n".join(
        f"    {variable} = inputs[:, {index}]"
        for index, variable in enumerate(variables)
    )
    text = replace_once(
        text,
        r"(?ms)(    outputs = np\.asarray\(\n.*?"
        r"    \)\.reshape\(-1\)\n\n).*?(\n    output_variance = np\.var\(outputs\))",
        "\\g<1>" + assignments + "\\g<2>",
        "input-column assignments",
    )

    call_arguments = "\n".join(
        [*(f"                    {variable}," for variable in variables),
         "                    params,"]
    )
    text = replace_once(
        text,
        r"(?ms)(                prediction = equation\(\n).*?"
        r"(                \)\n)",
        "\\g<1>" + call_arguments + "\n\\g<2>",
        "equation call",
    )

    signature_arguments = "\n".join(
        [*(f"    {variable}: np.ndarray," for variable in variables),
         "    params: np.ndarray,"]
    )
    text = replace_once(
        text,
        r"(?ms)(@equation\.evolve\ndef equation\(\n).*?(\) -> np\.ndarray:)",
        "\\g<1>" + signature_arguments + "\n\\g<2>",
        "equation signature",
    )

    text = replace_once(
        text,
        r"(?m)^        target = f\([^\n]+\)$",
        f"        {target} = f({variable_csv}, params)",
        "equation docstring formula",
    )

    args_entries = "\n\n".join(
        f"        {variable}:\n            Independent variable."
        for variable in variables
    )
    args_block = (
        f"    Args:\n{args_entries}\n\n"
        "        params:\n"
        "            Dataset-specific numerical parameters."
    )
    text = replace_once(
        text,
        r"(?ms)^    Args:\n.*?^        params:\n"
        r"            Dataset-specific numerical parameters\.",
        args_block,
        "equation Args section",
    )

    formula_lines = [f"        params[0] * {variables[0]}"]
    formula_lines.extend(
        f"        + params[{index}] * {variable}"
        for index, variable in enumerate(variables[1:], start=1)
    )
    formula_lines.append(f"        + params[{len(variables)}]")
    initial_formula = "\n".join(formula_lines)

    text = replace_once(
        text,
        r"(?ms)^    y = \(\n.*?^    \)\n\n    return y\s*$",
        f"    y = (\n{initial_formula}\n    )\n\n    return y",
        "initial equation body",
    )

    return text.rstrip() + "\n"


def load_rows(excel_path: Path, sheet: str | int) -> pd.DataFrame:
    if not excel_path.is_file():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    dataframe = pd.read_excel(excel_path, sheet_name=sheet)
    dataframe.columns = [str(column).strip().lstrip("\ufeff") for column in dataframe]

    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe]
    if missing:
        raise ValueError(f"Excel is missing required columns: {', '.join(missing)}")

    return dataframe.loc[:, list(REQUIRED_COLUMNS)].dropna(how="all")


def generate_all(
    excel_path: Path,
    template_path: Path,
    output_dir: Path,
    sheet: str | int,
) -> int:
    if not template_path.is_file():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    template = template_path.read_text(encoding="utf-8-sig")
    dataframe = load_rows(excel_path, sheet)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    total = len(dataframe)

    for position, (_, row) in enumerate(dataframe.iterrows(), start=1):
        if pd.isna(row["ID"]):
            raise ValueError(f"Excel row {position + 1}: ID is empty")

        row_id = str(row["ID"]).strip().lstrip("\ufeff")
        if not row_id:
            raise ValueError(f"Excel row {position + 1}: ID is empty")
        if row_id in seen_ids:
            raise ValueError(f"Duplicate ID: {row_id}")
        if Path(row_id).name != row_id or any(char in row_id for char in '<>:"/\\|?*'):
            raise ValueError(f"Unsafe ID for a filename: {row_id!r}")

        if pd.isna(row["Target"]):
            raise ValueError(f"{row_id}: Target is empty")
        target = require_python_identifier(str(row["Target"]), "Target", row_id)
        variables = parse_independent_vars(row["IndependentVars"], row_id)

        rendered = render_prompt(template, target, variables)
        destination = output_dir / f"{row_id}.txt"
        destination.write_text(rendered, encoding="utf-8", newline="\n")
        seen_ids.add(row_id)

        print(
            f"[{position:>{len(str(total))}}/{total}] "
            f"generated {destination.name}: target={target}, "
            f"variables={', '.join(variables)}",
            flush=True,
        )

    print(f"Done: generated {total} prompt files in {output_dir.resolve()}")
    return total


def main() -> None:
    args = parse_arguments()
    sheet: str | int = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)

    generate_all(
        excel_path=args.excel,
        template_path=args.template,
        output_dir=args.output_dir,
        sheet=sheet,
    )


if __name__ == "__main__":
    main()
