# Third-party sources

Vendored, unmodified, for the workbench (`tools/rmwork/`, `make workbench`, `docs/workbench_plan.md`).
Nothing else in the repository uses them. To bump one, replace its directory from the new pin and
update this table.

| Directory | Project | Pin | Licence |
|---|---|---|---|
| `imgui/` | Dear ImGui, docking branch (core files, GLFW and OpenGL 3 backends) | tag `v1.92.9-docking`, commit `9b4eb24cee2071e61dc1f9ef3e5228097cdde720` | MIT (`imgui/LICENSE.txt`) |
| `implot/` | ImPlot | tag `v0.17`, commit `4707b245fbcd69075b1a8a74fa8d2435561b3134` | MIT (`implot/LICENSE`) |
| `stb/` | `stb_image.h`, `stb_image_write.h` | commit `9600522de53973336fa033a289858a4354cbb04c` | public domain / MIT (in each header) |
| `fonts/jetbrains-mono/` | JetBrains Mono Regular, Medium | tag `v2.304` | SIL OFL 1.1 (`OFL.txt`) |
| `fonts/fira-code/` | Fira Code Regular, Medium | release `6.2` (`Fira_Code_v6.2.zip`) | SIL OFL 1.1 (`LICENSE`) |
| `fonts/ibm-plex-mono/` | IBM Plex Mono Regular, Medium | tag `v6.4.0` | SIL OFL 1.1 (`LICENSE.txt`) |
| `fonts/source-code-pro/` | Source Code Pro Regular, Medium | branch `release`, commit `803b7e23ec97ae58b6232ea76519a76d428ba268` | SIL OFL 1.1 (`LICENSE.md`) |
| `fonts/cascadia-mono/` | Cascadia Mono Regular, SemiBold (static) | release `v2407.24` (`CascadiaCode-2407.24.zip`) | SIL OFL 1.1 (`LICENSE`) |
| `fonts/hack/` | Hack Regular, Bold | tag `v3.003` | MIT, with Bitstream Vera / DejaVu terms (`LICENSE.md`) |
| `fonts/inter/` | Inter Regular, Medium (static, `extras/ttf`) | release `v4.1` (`Inter-4.1.zip`) | SIL OFL 1.1 (`LICENSE.txt`) |
| `fonts/ibm-plex-sans/` | IBM Plex Sans Regular, Medium | tag `v6.4.0` | SIL OFL 1.1 (`LICENSE.txt`) |
