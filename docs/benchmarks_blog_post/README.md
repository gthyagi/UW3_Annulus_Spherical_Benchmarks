# Stokes Benchmark Technical Note

The source is [blog_annulus_spherical_stokes_benchmarks.md](blog_annulus_spherical_stokes_benchmarks.md).
The existing filename is retained for incoming links.

To render it from the repository root, with Pandoc installed:

```bash
bash docs/benchmarks_blog_post/render_technical_note.sh
```

The command creates `blog_annulus_spherical_stokes_benchmarks.html` beside
the source. Open that file in a modern browser. Figures and CSS are embedded,
and equations use native MathML, so the preview needs no server or external
math-rendering service. The generated HTML is ignored by Git.

Use `$...$` for inline mathematics and `$$` on separate lines for display
equations. Leave a blank line before and after each display. Use `aligned`
for multiline equations and keep mathematical prose outside raw HTML
paragraphs. Figure captions remain Markdown inside blank-line-separated
centred containers so their inline mathematics is parsed.

The spacing, justification, and figure-caption styling of the local preview
are controlled by [technical_note.css](technical_note.css). GitHub applies
its own stylesheet, so those layout settings cannot be enforced in its
Markdown view. The equation delimiters are supported by
[GitHub's mathematical-expression syntax](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions).
