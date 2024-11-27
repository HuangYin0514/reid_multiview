import torch

if __name__ == "__main__":
    print("main")

    matrix = torch.range(1, 9).view(3, 3)
    print("Original matrix:\n", matrix)

    bs = matrix.size(0)
    mask = ~torch.eye(bs, dtype=torch.bool)
    non_diag_elements = matrix[mask]
    print("Non-diagonal elements:\n", non_diag_elements)

    bs = matrix.size(0)
    mask = torch.eye(bs, dtype=torch.bool)
    non_diag_elements = matrix[mask]
    print("diagonal elements:\n", non_diag_elements)
