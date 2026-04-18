"""Indicator search & configuration dialog (PyQt6)."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from wickly.indicators import (
    IndicatorSpec,
    categories,
    list_indicators,
)


def _is_dark(bg_hex: str) -> bool:
    """Heuristic: return True when *bg_hex* is a dark colour."""
    c = QColor(bg_hex)
    return c.lightnessF() < 0.45


class IndicatorSearchDialog(QDialog):
    """A searchable dialog for selecting and configuring indicators.

    Signals
    -------
    indicatorAdded(str, dict)
        Emitted when the user clicks *Add*.  Payload is
        ``(indicator_name, params_dict)``.
    indicatorRemoved(str)
        Emitted when the user clicks *Remove* on an active indicator.
        Payload is the ``indicator_id`` (unique key assigned by the chart).
    """

    indicatorAdded = pyqtSignal(str, dict)
    indicatorRemoved = pyqtSignal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        style: dict[str, Any] | None = None,
        active_indicators: list[tuple[str, str, dict]] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        parent : QWidget or None
        style : dict
            Chart style dict (used for theming).
        active_indicators : list of (indicator_id, display_label, params)
            Currently active indicators on the chart.
        """
        super().__init__(parent)
        self.setWindowTitle("Indicators")
        self.setMinimumSize(420, 500)
        self.resize(460, 560)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        self._style = style or {}
        self._active = list(active_indicators or [])
        self._selected_spec: IndicatorSpec | None = None
        self._param_widgets: dict[str, QSpinBox | QDoubleSpinBox] = {}

        self._build_ui()
        self._apply_theme()
        self._populate_list()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # --- active indicators section ---
        if self._active:
            active_label = QLabel("Active Indicators")
            active_label.setObjectName("sectionLabel")
            root.addWidget(active_label)
            for ind_id, label, _params in self._active:
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                lbl = QLabel(label)
                lbl.setObjectName("activeLabel")
                row.addWidget(lbl, stretch=1)
                btn = QPushButton("Remove")
                btn.setObjectName("removeBtn")
                btn.setFixedWidth(64)
                btn.clicked.connect(lambda checked, uid=ind_id: self._on_remove(uid))
                row.addWidget(btn)
                root.addLayout(row)
            # separator
            sep = QLabel("")
            sep.setFixedHeight(1)
            sep.setObjectName("separator")
            root.addWidget(sep)

        # --- search bar ---
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search indicators…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._filter_list)
        root.addWidget(self._search)

        # --- category tabs (simple QPushButton row) ---
        cat_row = QHBoxLayout()
        cat_row.setSpacing(4)
        self._cat_buttons: list[QPushButton] = []
        all_btn = QPushButton("All")
        all_btn.setCheckable(True)
        all_btn.setChecked(True)
        all_btn.setObjectName("catBtn")
        all_btn.clicked.connect(lambda: self._select_category(None))
        cat_row.addWidget(all_btn)
        self._cat_buttons.append(all_btn)
        for cat in categories():
            btn = QPushButton(cat)
            btn.setCheckable(True)
            btn.setObjectName("catBtn")
            btn.clicked.connect(lambda checked, c=cat: self._select_category(c))
            cat_row.addWidget(btn)
            self._cat_buttons.append(btn)
        cat_row.addStretch()
        root.addLayout(cat_row)

        # --- indicator list ---
        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_selection_changed)
        root.addWidget(self._list, stretch=1)

        # --- parameter panel (scrollable) ---
        self._param_area = QScrollArea()
        self._param_area.setWidgetResizable(True)
        self._param_area.setFixedHeight(0)  # hidden initially
        self._param_container = QWidget()
        self._param_layout = QVBoxLayout(self._param_container)
        self._param_layout.setContentsMargins(4, 4, 4, 4)
        self._param_layout.setSpacing(4)
        self._param_area.setWidget(self._param_container)
        root.addWidget(self._param_area)

        # --- add button ---
        self._add_btn = QPushButton("Add Indicator")
        self._add_btn.setObjectName("addBtn")
        self._add_btn.setEnabled(False)
        self._add_btn.clicked.connect(self._on_add)
        root.addWidget(self._add_btn)

        self._current_category: str | None = None

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        bg = self._style.get("bg_color", "#ffffff")
        dark = _is_dark(bg)

        if dark:
            base = "#1e1e2f"
            surface = "#282840"
            text = "#d4d4d4"
            muted = "#888"
            accent = "#4fc3f7"
            border = "#3a3a50"
            hover = "#333350"
            danger = "#ef5350"
        else:
            base = "#f8f8fa"
            surface = "#ffffff"
            text = "#222"
            muted = "#888"
            accent = "#1976d2"
            border = "#d4d4d4"
            hover = "#e8e8ee"
            danger = "#d32f2f"

        self.setStyleSheet(f"""
            QDialog {{
                background: {base};
                color: {text};
            }}
            QLabel {{
                color: {text};
                font-size: 12px;
            }}
            QLabel#sectionLabel {{
                font-weight: bold;
                font-size: 13px;
                padding-bottom: 2px;
            }}
            QLabel#activeLabel {{
                font-size: 12px;
            }}
            QLabel#separator {{
                background: {border};
            }}
            QLineEdit {{
                background: {surface};
                color: {text};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 6px 8px;
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border: 1px solid {accent};
            }}
            QListWidget {{
                background: {surface};
                color: {text};
                border: 1px solid {border};
                border-radius: 4px;
                font-size: 12px;
                outline: none;
            }}
            QListWidget::item {{
                padding: 5px 8px;
                border-radius: 3px;
            }}
            QListWidget::item:hover {{
                background: {hover};
            }}
            QListWidget::item:selected {{
                background: {accent};
                color: #fff;
            }}
            QPushButton#catBtn {{
                background: {surface};
                color: {text};
                border: 1px solid {border};
                border-radius: 10px;
                padding: 3px 10px;
                font-size: 11px;
            }}
            QPushButton#catBtn:checked {{
                background: {accent};
                color: #fff;
                border-color: {accent};
            }}
            QPushButton#addBtn {{
                background: {accent};
                color: #fff;
                border: none;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton#addBtn:disabled {{
                background: {muted};
            }}
            QPushButton#removeBtn {{
                background: {danger};
                color: #fff;
                border: none;
                border-radius: 3px;
                padding: 3px 6px;
                font-size: 11px;
            }}
            QScrollArea {{
                border: none;
                background: transparent;
            }}
            QSpinBox, QDoubleSpinBox {{
                background: {surface};
                color: {text};
                border: 1px solid {border};
                border-radius: 3px;
                padding: 3px 6px;
            }}
        """)

    # ------------------------------------------------------------------
    # List population / filtering
    # ------------------------------------------------------------------

    def _populate_list(self) -> None:
        self._list.clear()
        specs = list_indicators(self._current_category)
        query = self._search.text().strip().lower()
        for spec in specs:
            text = f"{spec.display_name}"
            if query and query not in spec.display_name.lower() and query not in spec.name.lower():
                continue
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, spec.name)
            self._list.addItem(item)

    def _filter_list(self, text: str) -> None:
        self._populate_list()

    def _select_category(self, category: str | None) -> None:
        self._current_category = category
        for btn in self._cat_buttons:
            is_all = btn.text() == "All"
            btn.setChecked(
                (category is None and is_all) or btn.text() == category
            )
        self._populate_list()

    # ------------------------------------------------------------------
    # Selection → parameter panel
    # ------------------------------------------------------------------

    def _on_selection_changed(self, current: QListWidgetItem | None, _prev: QListWidgetItem | None) -> None:
        # Clear old params
        while self._param_layout.count():
            child = self._param_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self._param_widgets.clear()
        self._selected_spec = None
        self._add_btn.setEnabled(False)

        if current is None:
            self._param_area.setFixedHeight(0)
            return

        from wickly.indicators import get_indicator
        name = current.data(Qt.ItemDataRole.UserRole)
        spec = get_indicator(name)
        self._selected_spec = spec
        self._add_btn.setEnabled(True)

        if not spec.params:
            self._param_area.setFixedHeight(0)
            return

        # Build param widgets
        for ps in spec.params:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(ps.label)
            lbl.setFixedWidth(100)
            row.addWidget(lbl)

            if isinstance(ps.default, float) or isinstance(ps.step, float):
                spin = QDoubleSpinBox()
                spin.setDecimals(2)
                spin.setMinimum(float(ps.min_val))
                spin.setMaximum(float(ps.max_val))
                spin.setSingleStep(float(ps.step))
                spin.setValue(float(ps.default))
            else:
                spin = QSpinBox()
                spin.setMinimum(int(ps.min_val))
                spin.setMaximum(int(ps.max_val))
                spin.setSingleStep(int(ps.step))
                spin.setValue(int(ps.default))

            row.addWidget(spin, stretch=1)
            self._param_widgets[ps.name] = spin

            wrapper = QWidget()
            wrapper.setLayout(row)
            self._param_layout.addWidget(wrapper)

        n_params = len(spec.params)
        self._param_area.setFixedHeight(min(n_params * 36 + 16, 140))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_add(self) -> None:
        if self._selected_spec is None:
            return
        params: dict[str, Any] = {}
        for ps in self._selected_spec.params:
            widget = self._param_widgets.get(ps.name)
            if widget is not None:
                params[ps.name] = widget.value()
        self.indicatorAdded.emit(self._selected_spec.name, params)
        self.accept()

    def _on_remove(self, indicator_id: str) -> None:
        self.indicatorRemoved.emit(indicator_id)
        # Remove from local list and rebuild UI
        self._active = [(uid, lbl, p) for uid, lbl, p in self._active if uid != indicator_id]
        self.accept()
