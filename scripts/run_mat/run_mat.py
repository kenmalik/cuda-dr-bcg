from tempfile import NamedTemporaryFile
import sys
import subprocess
from scipy.io import loadmat


def run_mat(file_name: str, command_args: list[str], matrix_name="A", problem_name="Problem") -> subprocess.CompletedProcess:
    data = loadmat(file_name, struct_as_record=False, squeeze_me=True)

    problem = data.get(problem_name)
    if problem is None:
        raise AttributeError(f"'{problem_name}' field not found in {file_name}")

    try:
        mat = getattr(problem, matrix_name)
    except AttributeError as e:
        raise AttributeError(f"Matrix with name {matrix_name} not found in problem") from e

    with NamedTemporaryFile() as tmp:
        print(f"Converting {file_name} to temp file", file=sys.stderr)
        dense = mat.toarray()
        dense.T.tofile(tmp.name)

        for i, arg in enumerate(command_args):
            if arg == "{file}":
                command_args[i] = tmp.name
            elif arg == "{m}":
                command_args[i] = str(problem.A.shape[0])
            elif arg == "{n}":
                command_args[i] = str(problem.A.shape[1])

        print(f"Running the following command: '{' '.join(command_args)}'", file=sys.stderr)

        res = subprocess.run(
            command_args, check=True, capture_output=True, text=True
        )
        return res


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_mat.py [MAT_FILE] [COMMAND_ARGS...]")
        exit(1)

    file_name = sys.argv[1]
    command_args = sys.argv[2:]

    try:
        res = run_mat(file_name, command_args)
        print(res.stdout)
    except FileNotFoundError as e:
        print(f"File '{file_name}' not found", file=sys.stderr)
    except AttributeError as e:
        print(e, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}\n")
        print(e.stdout)
        print(e.stderr, file=sys.stderr)
