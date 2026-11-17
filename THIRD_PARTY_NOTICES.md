# Third-party notices

gputaskman is MIT licensed (see [LICENSE](LICENSE)). It does not copy or vendor any third-party source code.
The packages below are installed alongside it as dependencies, each under its own license, which ships with
that package. The portable Windows ZIP (`tools/build_portable.py`) bundles its dependencies and includes their
license texts in the ZIP itself.

## Runtime dependencies (`gputaskman`)

| Package | License | Project |
|---|---|---|
| nvidia-ml-py | BSD-3-Clause | https://pypi.org/project/nvidia-ml-py/ |
| psutil | BSD-3-Clause | https://github.com/giampaolo/psutil |
| textual | MIT | https://github.com/Textualize/textual |

## Desktop app extra (`gputaskman[app]`, the `gputaskman-app` window)

| Package | License | Project |
|---|---|---|
| pywebview | BSD-3-Clause | https://github.com/r0x0r/pywebview |
| pythonnet (Windows) | MIT | https://github.com/pythonnet/pythonnet |
| clr-loader (Windows) | MIT | https://github.com/pythonnet/clr-loader |
| proxy-tools | MIT | https://github.com/jtushman/proxy_tools |
| bottle | MIT | https://github.com/bottlepy/bottle |
| typing-extensions | PSF-2.0 | https://github.com/python/typing_extensions |
| PyObjC (macOS: pyobjc-core and Cocoa, Quartz, Security, UniformTypeIdentifiers, WebKit frameworks) | MIT | https://github.com/ronaldoussoren/pyobjc |
| QtPy (OpenBSD) | MIT | https://github.com/spyder-ide/qtpy |

On Linux pywebview draws with GTK or Qt, which come from the system's own packages, not from this project.

On Windows the window is drawn by Microsoft Edge WebView2, which is part of Windows 11 and is not
redistributed by this project.

The desktop app's icons are drawn inline as SVG in `src/gputaskman/desktop/ui/index.html` for this project;
no icon set or font is bundled (it uses the system's Segoe UI).
