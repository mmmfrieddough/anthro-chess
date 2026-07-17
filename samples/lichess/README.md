# Lichess Standard Export Sample

`standard-export-sample.pgn` is the sample game published on the
[Lichess open database](https://database.lichess.org/) page for its standard
rated PGN exports. Lichess releases those database exports under CC0, including
permission to modify and redistribute them.

The checked-in file keeps the published PGN content intact and lets ordinary
tests and the documented data-preparation command run without network access.
The source selection and license identifier used in generated manifests live
in `configs/data/lichess-sample.toml`.
