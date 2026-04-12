"""Tests for wickly.styles module."""

import pytest
from wickly.styles import available_styles, _get_style, make_style


class TestAvailableStyles:
    def test_returns_list(self):
        result = available_styles()
        assert isinstance(result, list)

    def test_default_in_list(self):
        assert "default" in available_styles()

    def test_all_builtins_present(self):
        expected = {"default", "charles", "mike", "yahoo", "classic", "nightclouds"}
        assert expected.issubset(set(available_styles()))


class TestGetStyle:
    def test_none_returns_default(self):
        s = _get_style(None)
        assert s["up_color"] == "#26a69a"

    def test_by_name(self):
        s = _get_style("charles")
        assert s["up_color"] == "#006340"

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown style"):
            _get_style("nonexistent_style")

    def test_dict_passthrough(self):
        custom = {"up_color": "#ff0000"}
        s = _get_style(custom)
        assert s["up_color"] == "#ff0000"
        # other keys should still be filled from default
        assert "down_color" in s


class TestMakeStyle:
    def test_default_base(self):
        s = make_style()
        assert s["up_color"] == "#26a69a"

    def test_override(self):
        s = make_style(up_color="#112233")
        assert s["up_color"] == "#112233"

    def test_base_style(self):
        s = make_style(base_mpf_style="mike")
        assert s["bg_color"] == "#0a0a23"

    def test_mavcolors(self):
        colors = ["red", "blue"]
        s = make_style(mavcolors=colors)
        assert s["mavcolors"] == colors
