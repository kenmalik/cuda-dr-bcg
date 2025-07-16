from run_mat import run_mat
from pathlib import Path
from subprocess import CalledProcessError
from typing import NamedTuple
from enum import Enum
import sys
import csv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("directory")
parser.add_argument("command_args", nargs="+")
parser.add_argument("-o", "--output", help="CSV file to output results to")
parser.add_argument(
    "-l", "--limit", help="Limit for the amount of files to run", type=int
)
parser.add_argument(
    "--enumerate",
    help="Only output the files to be ran on",
    action=argparse.BooleanOptionalAction,
)


class Status(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    NOT_RAN = "NOT_RAN"
    def __str__(self):
        return f"{self.name}"


class Row(NamedTuple):
    file: str
    status: Status
    return_code: int | None
    iterations: int


def run_files(
    directory_path: Path,
    command_args_template: list[str],
    limit: int | None = None,
    enumerate_files: bool = False,
) -> list[Row]:
    mat_files = list(directory_path.glob("*.mat"))

    if limit is None:
        limit = len(mat_files)

    results = []
    for i, file_name in enumerate(mat_files):
        if i == limit:
            break

        file = file_name.parts[-1]
        status = Status.NOT_RAN
        return_code = -1
        iterations = 0

        command_args = command_args_template.copy()
        print(f"{file}: '{' '.join(command_args)}'", file=sys.stderr)

        try:
            if not enumerate_files:
                res = run_mat(str(file_name), command_args)
                return_code = res.returncode
                iterations = int(res.stdout)
                if return_code == 0:
                    status = Status.PASSED
        except FileNotFoundError as e:
            print(f"File '{file_name}' not found", file=sys.stderr)
            status = Status.FAILED
        except AttributeError as e:
            print(e, file=sys.stderr)
            status = Status.FAILED
        except CalledProcessError as e:
            print(f"An error occurred for file {file}: {e}\n", file=sys.stderr)
            return_code = e.returncode
            status = Status.FAILED
            print(e.stderr, file=sys.stderr)

        row = Row(file, status, return_code, iterations)
        print(f"Result: {row}", file=sys.stderr)
        results.append(row)

    return results


def main():
    args = parser.parse_args()

    directory_path = Path(args.directory)
    if not directory_path.is_dir():
        print(f"{args.directory} is not a directory", file=sys.stderr)
        return

    res = run_files(
        directory_path,
        args.command_args,
        limit=args.limit,
        enumerate_files=args.enumerate,
    )

    if args.output:
        out_file = args.output
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(Row._fields)
            writer.writerows(res)
        print(f"Outputted to {out_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
