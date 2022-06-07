# Research on the quality of localization of movie titles

[Notebook]()  

`Multilingual Sentence Embeddings` `Text Similarity`  
The goal of this research is to find out:

* How similar Russian titles and original titles are in general?
* Can we split dissimilar pairs (`Russian title`::`original title`) into groups by root cause?

Dataset: Cropped dataset `top-1000 popular movies in Russia` (only 3 columns). Full dataset will be released later.  

* Dimensions:
  * 984 rows
* Features:
  * Text columns:
    * `russian_title`
    * `original_title`
    * `country`

Implementation includes:

* **Exploratory Data Analysis**
  * Exploring dataset
  * Cleaning dataset
    * Removing year from the `russian_title` column
    * Removing duplicates
    * Working with missing values
* **Analysis**:
  * Computing several similarity scores for different multilingual embeddings using [Sentence-Transformers](https://www.sbert.net/docs/pretrained_models.html#multi-lingual-models):
    * distiluse-base-multilingual-cased-v1
    * distiluse-base-multilingual-cased-v2
    * paraphrase-multilingual-MiniLM-L12-v2
    * paraphrase-multilingual-mpnet-base-v2
    * LaBSE
  * Calculating single similarity score for every pair `russian_title`::`original_title`
  * Analyzing results
    * Computing descriptive statistics for similarity score vector
    * Plotting distribution plot
    * Splitting dissimilar pairs `russian_title`::`original_title` into groups with reasoning

## TODO

* Make good visualizations:
  * [Visualizing Embeddings With t-SNE](https://www.kaggle.com/code/colinmorris/visualizing-embeddings-with-t-sne/notebook)
* Add into comparison deleted titles  
For languages, where transcription is used instead of original title (hieroglyphs)
* Add new data (top-1000 series titles)
