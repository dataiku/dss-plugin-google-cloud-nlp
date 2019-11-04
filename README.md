# Google Cloud Natural Language plugin

This plugin provides leverages various Google Cloud AI APIs for Natural Language Processing (NLP).

## Pre-requisites

- A service account key from a GCP project with access to the Cloud Natural Language and Cloud Vision APIs

## Recipes

The plugin offers a suite of recipes:

* **text content classification**: the recipe calls the [`classifyText`](https://cloud.google.com/natural-language/docs/classifying-text) method of the Cloud Natural Language API and returns a list of inferred categories, each with an associated
confidence score. For each category of a single input data point, a new row is generated in the output dataset.
* **entity analysis**: the recipe calls the [`analyzeEntities`](https://cloud.google.com/natural-language/docs/reference/rest/v1/documents/analyzeEntities) method of the Cloud Natural Language API and returns a list of inferred entities. For each entity detected in a single input data point, a new row is generated, containing informations about:
    - the name and type of the entity
    - the *salience*, which quantifies the importance of the entity in the overall text
    - the number and positions of the occurences of the entity
    - optionally, additional metadata (e.g. link to Wikipedia article)
* **sentiment analysis**: the recipe calls the [`analyzeSentiment`](https://cloud.google.com/natural-language/docs/analyzing-sentiment) method of the Cloud Natural Language API and returns an inferred measurement of the emotional opinion within the text. For each input data point, the output dataset completes it with:
    - a sentiment score between 0 and 1 reflects the emotion contained in the text, ranging from negative to positive sentiment
    - a magnitude score, which illustrates the quantity of opinionated content in the overall text
    - a breakdown version of the input text split into sentences, each with its individual sentiment score and magnitude

## Authentication

Accessing the Cloud API endpoints requires a *service account key* that you need to generate in your GCP project. Once this is done, you will have two options to enforce authentication using that key:

- **application default credentials**: on the DSS server you need to create an environment variable called `GOOGLE_APPLICATION_CREDENTIALS` which contains the absolute path to your service account key.
- **service account key**: manually enter the absolute path to the key within the plugin interface

## External resources

- [Cloud Natural Language AI doc](https://cloud.google.com/natural-language/docs/)
