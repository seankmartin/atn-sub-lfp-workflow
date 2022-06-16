from pathlib import Path

import typer
from skm_pyutils.table import df_from_file, df_to_file, list_to_df

app = typer.Typer()


def convert_name_to_type(name):
    p1 = name.split("_")[0]
    if p1 in ["Control", "Muscimol"]:
        return p1
    return "Spatial" if p1 == "S" else "Non-Spatial"


@app.command()
def main(filepath: str):
    units_dict = {}
    df = df_from_file(filepath)
    for i, row in df.iterrows():
        f_path = Path(row["Directory"]) / row["Filename"]
        if f_path not in units_dict:
            units_dict[f_path] = []
        group = row["Group"]
        unit = row["Unit"]
        units_dict[f_path].append(
            (f"TT{group}_U{unit}", convert_name_to_type(row["class"]))
        )

    l = []
    for k, v in units_dict.items():
        units = [x[0] for x in v]
        types = [x[1] for x in v]
        l.append([str(k.parent), str(k.name), units, types])
    headers = ["directory", "filename", "units", "unit_types"]
    df = list_to_df(l, headers=headers)
    filename = Path(filepath)
    filename = filename.with_name(f"{filename.stem}--converted.csv")
    df_to_file(df, filename)


if __name__ == "__main__":
    app()
