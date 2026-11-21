# models/

Model exports packaged as zip archives (e.g. `league26-24h-model.zip`). Drop an archive here as it is,
without unzipping it: the workbench reads it directly and lists every `*.ts.pt` / `*.pt2` inside in
Game → New Game vs Model as `<archive>: <run>/<file>`. When a game starts, the chosen export is
extracted once into the workbench cache (`~/.cache/robomage/workbench/models/`) for the engine.

A path typed into the dialog may name a member as `models/<archive>.zip!<member>`, or the bare
`models/<archive>.zip` when the archive holds one export.

Everything here but this README is ignored by git.
