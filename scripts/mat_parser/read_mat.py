import sys
import numpy as np
from scipy.io import loadmat
import scipy.sparse

def get_matrix(data, matrix_name='A'):
    """Extract the specified matrix ('A', 'M', 'B', 'C', etc.) from a UF-style .mat file."""
    if 'Problem' not in data:
        raise ValueError("MAT file does not contain a 'Problem' key.")

    problem = data['Problem']  # Already squeezed by loadmat

    if matrix_name == 'A':
        matrix = problem.A
    else:
        aux = problem.aux
        if not hasattr(aux, matrix_name):
            raise ValueError(f"Matrix '{matrix_name}' not found in 'aux' field.")
        matrix = getattr(aux, matrix_name)

    if not scipy.sparse.issparse(matrix):
        raise TypeError(f"Expected sparse matrix for '{matrix_name}', got {type(matrix)}.")

    return matrix

def save_matrix(matrix, out_file, fmt='csv'):
    """Save the matrix in the desired format: 'csv' or 'bin'."""
    dense = matrix.toarray()

    if fmt == 'csv':
        np.savetxt(out_file, dense, delimiter=',')
        print(f"Matrix saved as CSV: {out_file}")
    elif fmt == 'bin':
        # Write in column-major order to match MATLAB/CUDA layout
        dense.T.tofile(out_file)
        print(f"Matrix saved as binary (column-major): {out_file}")
    else:
        raise ValueError("Unsupported format. Use 'csv' or 'bin'.")

def main():
    if len(sys.argv) != 4:
        print("Usage: python extract_matrix_from_mat.py input.mat matrix_name output.{csv|bin}")
        print("Example: python extract_matrix_from_mat.py LFAT5.mat A A.csv")
        sys.exit(1)

    mat_file = sys.argv[1]
    matrix_name = sys.argv[2]
    out_file = sys.argv[3]
    fmt = sys.argv[3].lower().split('.')[-1]

    data = loadmat(mat_file, struct_as_record=False, squeeze_me=True)
    matrix = get_matrix(data, matrix_name)
    save_matrix(matrix, out_file, fmt)

if __name__ == "__main__":
    main()
