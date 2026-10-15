# Dataset Provenance

## Source

- Dataset: [Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment%2Blabelled%2Bsentences)
- DOI: [10.24432/C57604](https://doi.org/10.24432/C57604)
- Archive URL: https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip
- Contents: 3,000 sentences from IMDb, Yelp, and Amazon, with binary labels (`1` positive, `0` negative); 500 examples per class and source.
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- Dataset citation: Kotzias, D. (2015). *Sentiment Labelled Sentences* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C57604.
- Associated paper: Kotzias et al. (2015), “From Group to Individual Labels Using Deep Features,” KDD 2015. UCI requests citation when the dataset is used.

## Fetching

The archive will be downloaded by `scripts/download_data.py` into `data/raw/`. Until that script is implemented, fetch the ZIP from the archive URL above and extract the three `*_labelled.txt` files into `data/raw/`. Do not commit the downloaded files; `.gitignore` excludes this directory. The training loader will validate the tab-separated text/label format and map the values to binary labels.

This small, curated dataset is suitable for a transparent baseline, not a claim of production-grade sentiment coverage. Its source domains and collection period constrain generalization.
