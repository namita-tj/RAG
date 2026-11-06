import pytest

from tablerag.normalize import clean, numbers_in, parse_number, parse_range, same_value, to_base, unit_in_header


@pytest.mark.parametrize(
    ("text", "value"), [("200", 200.0), ("1,5", 1.5), ("1,234", 1234.0), ("12 bar", 12.0), ("-3.5", -3.5)]
)
def test_parse_number(text, value):
    assert parse_number(text) == value


@pytest.mark.parametrize("text", ["900 - 1500", "n/a", "", "4 and 5"])
def test_parse_number_refuses_what_is_not_one_number(text):
    assert parse_number(text) is None


def test_parse_range_handles_dashes_and_units():
    assert parse_range("900 – 1500") == (900.0, 1500.0, "")
    assert parse_range("0.15 to 30 MHz") == (0.15, 30.0, "MHz")
    assert parse_range("1500 - 900") is None  # reversed is not a range
    assert parse_range("Stage 3") is None


def test_units_convert_within_a_quantity_only():
    assert to_base(1.5, "kHz") == ("frequency", 1500.0)
    assert to_base(2, "bar") == ("pressure", 200000.0)
    assert to_base(3, "furlongs") is None
    assert unit_in_header("Max. pressure [bar]") == "bar" and unit_in_header("Stage") == ""


def test_same_value_is_exact_unless_asked():
    assert same_value("200", "200.0") and not same_value("200", "200.1")
    assert same_value("100", "100.4", rel_tol=0.01)
    assert same_value("Stage  3", "stage 3")


def test_clean_and_numbers_in():
    assert clean("a–b  c") == "a-b c"
    assert numbers_in("between 900 and 1,500 rpm") == [900.0, 1500.0]
