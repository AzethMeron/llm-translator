"""A minimal fake *carrier adapter*, for exercising the adapter protocol in tests.

The real overlying adapters (subtitles, game engines, documents) are deliberately not part
of this repository -- this library is the translator alone. But the translator's downstream
boundary, :mod:`transunit.adapter`, still needs to be tested: that ``load_adapter`` finds a
package, that the protocol is satisfied, and that the translator can be handed an adapter's
``sanitize_payload``. This package is the smallest thing that stands in for a real one.
"""
