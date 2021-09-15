# The testing loop I used

This command uses the latest version to annoate the dataset

``` sh
cargo run -- explain -i data/Wine/winequality-src-both.csv -r data/Wine/winequality-tsne-3d.csv data/Wine/annotations-3d.csv
cargo run -- explain -i data/Wine/winequality-src-both.csv -r data/Wine/winequality-tsne-2d.csv -b data/Wine/annotations-2d.csv
```

# Run the viewer
``` sh
# cube 2d pca
cargo run --release -- view -i data/cube/cube.csv --r2d data/cube/reduced-cube-pca-2d.csv

# winequality 2d lamp
cargo run --release -- view -i data/Wine/winequality-src-both.csv --r2d data/Wine/winequality-lamp-both-2d.csv

# winequality 3d tsne
cargo run --release -- view -i data/Wine/winequality-src-both.csv --r3d data/Wine/winequality-tsne-3d.csv

# Both winequality
cargo run --release -- view -i data/Wine/winequality-src-both.csv --r2d data/Wine/winequality-lamp-both-2d.csv --r3d data/Wine/winequality-tsne-3d.csv
```