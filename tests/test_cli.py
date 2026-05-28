from tinyedgebench.cli import build_wizard_case


def test_wizard_matmul_case_from_mocked_input() -> None:
    answers = iter(
        [
            "matmul",
            "fp32,int8_sim",
            "4",
            "5",
            "6",
        ]
    )

    case = build_wizard_case(input_func=lambda _: next(answers), print_func=lambda _: None)

    assert case.operator == "matmul"
    assert case.precision_modes == ["fp32", "int8_sim"]
    assert case.matrix_m == 4
    assert case.matrix_k == 5
    assert case.matrix_n == 6


def test_wizard_conv2d_case_from_mocked_input() -> None:
    answers = iter(
        [
            "conv2d",
            "fp32,shift_only",
            "1,3,8,8",
            "3,3",
            "1",
            "1",
            "4",
        ]
    )

    case = build_wizard_case(input_func=lambda _: next(answers), print_func=lambda _: None)

    assert case.operator == "conv2d"
    assert case.input_shape == (1, 3, 8, 8)
    assert case.output_channels == 4
    assert case.kernel_size == (3, 3)
    assert case.precision_modes == ["fp32", "shift_only"]
