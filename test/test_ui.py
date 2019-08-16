import unittest
import unittest.mock
from forest import layer
import bokeh.models


class TestComponent(unittest.TestCase):
    def test_mute_glyph_on_source_change(self):
        figure = bokeh.plotting.figure()
        source_factory = layer.SourceFactory()
        glyph_factory = layer.GlyphFactory(
            source_factory,
            figure)
        visible_glyphs = layer.VisibleGlyphs(glyph_factory)
        visible_glyphs.on_change(None, None, "file.nc")
        visible_glyphs.on_change(None, None, "file.json")
        result = visible_glyphs.glyphs['line'].visible
        expect = False
        self.assertEqual(result, expect)

    def test_on_change_given_layer(self):
        figures = [
            bokeh.plotting.figure(),
            bokeh.plotting.figure(),
            bokeh.plotting.figure()
        ]
        menu = [
                ("option 1", "file.nc"),
                ("option 2", "file.json")]
        controls = layer.Controls(figures, menu)
        source_factory = layer.SourceFactory()
        controls.on_change(source_factory)(None, None, "file.json")
        result = source_factory.sources
        expect = {}
        self.assertEqual(result, expect)


class TestGlyphMap(unittest.TestCase):
    def test_mute_all(self):
        figure = bokeh.plotting.figure()
        renderer = figure.circle(x=[], y=[])
        glyph_map = layer.GlyphMap()
        glyph_map.insert("key", renderer)
        glyph_map.mute_all()
        self.assertEqual(renderer.visible, False)

    def test_mute_all_except(self):
        figure = bokeh.plotting.figure()
        renderers = [
                figure.circle(x=[], y=[]),
                figure.circle(x=[], y=[]),
                figure.circle(x=[], y=[])]
        glyph_map = layer.GlyphMap()
        glyph_map.insert("A", renderers[0])
        glyph_map.insert("B", renderers[1])
        glyph_map.insert("C", renderers[2])
        glyph_map.mute_all_except("B")
        self.assertEqual(renderers[0].visible, False)
        self.assertEqual(renderers[1].visible, True)
        self.assertEqual(renderers[2].visible, False)


class TestUI(unittest.TestCase):
    def setUp(self):
        self.ui = layer.UI()

    def test_add_remove_buttons(self):
        self.assertIsInstance(
                self.ui.layout, bokeh.layouts.Column)

    def test_layout_has_one_row(self):
        self.assert_is_row(self.ui.layout.children[0])

    def test_layout_has_add_button(self):
        row = self.ui.layout.children[-1]
        labels = []
        for child in row.children:
            if isinstance(child, bokeh.models.Button):
                labels.append(child.label)
        self.assertIn("Add", labels)

    def test_layout_has_remove_button(self):
        row = self.ui.layout.children[-1]
        labels = []
        for child in row.children:
            if isinstance(child, bokeh.models.Button):
                labels.append(child.label)
        self.assertIn("Remove", labels)

    def test_on_add_inserts_row_before_add_remove_row(self):
        ui = layer.UI()
        add_remove_id = ui.layout.children[0].id
        ui.on_add()
        self.assertEqual(len(ui.layout.children), 2)
        self.assertEqual(ui.layout.children[-1].id, add_remove_id)

    def test_on_add_inserts_row(self):
        self.ui.on_add()
        self.assert_is_row(self.ui.layout.children[0])

    def test_on_remove_reduces_buttons_by_one(self):
        expect = self.ui.layout.children[0].id
        self.ui.on_add()
        self.ui.on_remove()
        self.assertEqual(self.ui.layout.children[-1].id, expect)

    def test_on_add_emits_ui_add_action(self):
        listener = unittest.mock.Mock()
        self.ui.subscribe(listener)
        self.ui.on_add()
        dropdown = self.ui.layout.children[0].children[0]
        group = self.ui.layout.children[0].children[1]
        expect = layer.ui_add(dropdown, group)
        listener.assert_called_once_with(expect)

    def test_on_remove_emits_ui_remove_action(self):
        listener = unittest.mock.Mock()
        self.ui.on_add()
        self.ui.subscribe(listener)
        self.ui.on_remove()
        expect = layer.ui_remove()
        listener.assert_called_once_with(expect)

    def assert_is_row(self, child):
        self.assertIsInstance(child, bokeh.layouts.Row)
