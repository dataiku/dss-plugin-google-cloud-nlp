# -*- coding: utf-8 -*-
import logging
import pandas as pd

from enum import Enum
from typing import AnyStr, Dict, List, Union

from plugin_io_utils import (
    API_COLUMN_NAMES_DESCRIPTION_DICT,
    ErrorHandlingEnum,
    build_unique_column_names,
    generate_unique,
    safe_json_loads,
    move_api_columns_to_end,
)


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================


class EntityTypeEnum(Enum):
    ADDRESS = "Address"
    CONSUMER_GOOD = "Consumer good"
    DATE = "Date"
    EVENT = "Event"
    LOCATION = "Location"
    NUMBER = "Number"
    ORGANIZATION = "Organization"
    OTHER = "Other"
    PERSON = "Person"
    PHONE_NUMBER = "Phone number"
    PRICE = "Price"
    UNKNOWN = "Unknown"
    WORK_OF_ART = "Work of art"


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class GenericAPIFormatter:
    """
    Geric Formatter class for API responses:
    - initialize with generic parameters
    - compute generic column descriptions
    - apply format_row to dataframe
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        column_prefix: AnyStr = "api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        self.input_df = input_df
        self.column_prefix = column_prefix
        self.error_handling = error_handling
        self.api_column_names = build_unique_column_names(input_df, column_prefix)
        self.column_description_dict = {
            v: API_COLUMN_NAMES_DESCRIPTION_DICT[k]
            for k, v in self.api_column_names._asdict().items()
        }

    def format_row(self, row: Dict) -> Dict:
        return row

    def format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Formatting API results...")
        df = df.apply(func=self.format_row, axis=1)
        df = move_api_columns_to_end(df, self.api_column_names)
        logging.info("Formatting API results: Done.")
        return df


class SentimentAnalysisAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for Sentiment Analysis API responses:
    - make sure response is valid JSON
    - expand results to two score and magnitude columns
    - scale the score according to categorical or numerical rules
    - compute column descriptions
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        sentiment_scale: AnyStr = "ternary",
        column_prefix: AnyStr = "sentiment_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        super().__init__(input_df, column_prefix, error_handling)
        self.sentiment_scale = sentiment_scale
        self.sentiment_score_column = generate_unique(
            "score", input_df.keys(), self.column_prefix
        )
        self.sentiment_score_scaled_column = generate_unique(
            "score_scaled", input_df.keys(), column_prefix
        )
        self.sentiment_magnitude_column = generate_unique(
            "magnitude", input_df.keys(), column_prefix
        )
        self._compute_column_description()

    def _compute_column_description(self):
        self.column_description_dict[
            self.sentiment_score_column
        ] = "Sentiment score from the API in numerical format between -1 and 1"
        self.column_description_dict[
            self.sentiment_score_scaled_column
        ] = "Scaled sentiment score according to “Sentiment scale” parameter"
        self.column_description_dict[
            self.sentiment_magnitude_column
        ] = "Magnitude score indicating emotion strength (both positive and negative) between 0 and +Inf"

    def _scale_sentiment_score(
        self, score: float, sentiment_scale: AnyStr = "ternary"
    ) -> Union[AnyStr, float]:
        if sentiment_scale == "binary":
            return "negative" if score < 0 else "positive"
        elif sentiment_scale == "ternary":
            if score < -0.33:
                return "negative"
            elif score > 0.33:
                return "positive"
            else:
                return "neutral"
        elif sentiment_scale == "quinary":
            if score < -0.66:
                return "highly negative"
            elif score < -0.33:
                return "negative"
            elif score < 0.33:
                return "neutral"
            elif score < 0.66:
                return "positive"
            else:
                return "highly positive"
        elif sentiment_scale == "rescale_zero_to_one":
            return float((score + 1.0) / 2)
        else:
            return float(score)

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        sentiment = response.get("documentSentiment", {})
        sentiment_score = sentiment.get("score")
        magnitude_score = sentiment.get("magnitude")
        if sentiment_score is not None:
            row[self.sentiment_score_column] = float(sentiment_score)
            row[self.sentiment_score_scaled_column] = self._scale_sentiment_score(
                sentiment_score, self.sentiment_scale
            )
        else:
            row[self.sentiment_score_column] = None
            row[self.sentiment_score_scaled_column] = None
        if magnitude_score is not None:
            row[self.sentiment_magnitude_column] = float(magnitude_score)
        else:
            row[self.sentiment_magnitude_column] = None
        return row


class NamedEntityRecognitionAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for Named Entity Recognition API responses:
    - make sure response is valid JSON
    - expand results to multiple columns (one by entity type)
    - compute column descriptions
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        entity_types: List,
        minimum_score: float,
        column_prefix: AnyStr = "entity_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        super().__init__(input_df, column_prefix, error_handling)
        self.entity_types = entity_types
        self.minimum_score = float(minimum_score)
        self._compute_column_description()

    def _compute_column_description(self):
        for n, m in EntityTypeEnum.__members__.items():
            entity_type_column = generate_unique(
                "entity_type_" + n.lower(), self.input_df.keys(), self.column_prefix
            )
            self.column_description_dict[
                entity_type_column
            ] = "List of '{}' entities recognized by the API".format(str(m.value))

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        entities = response.get("entities", [])
        selected_entity_types = sorted([e.name for e in self.entity_types])
        for n in selected_entity_types:
            entity_type_column = generate_unique(
                "entity_type_" + n.lower(), row.keys(), self.column_prefix
            )
            row[entity_type_column] = [
                e.get("name")
                for e in entities
                if e.get("type", "") == n
                and float(e.get("salience", 0)) >= self.minimum_score
            ]
            if len(row[entity_type_column]) == 0:
                row[entity_type_column] = ""
        return row


class TextClassificationAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for Text Classification API responses:
    - make sure response is valid JSON
    - expand results to multiple columns (one by classification prediction)
    - compute column descriptions
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        num_categories: int = 1,
        column_prefix: AnyStr = "text_classif_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        super().__init__(input_df, column_prefix, error_handling)
        self.num_categories = num_categories
        self._compute_column_description()

    def _compute_column_description(self):
        for n in range(self.num_categories):
            category_column = generate_unique(
                "category_" + str(n + 1) + "_name",
                self.input_df.keys(),
                self.column_prefix,
            )
            confidence_column = generate_unique(
                "category_" + str(n + 1) + "_confidence",
                self.input_df.keys(),
                self.column_prefix,
            )
            self.column_description_dict[
                category_column
            ] = "Name of the category {} representing the document".format(str(n + 1))
            self.column_description_dict[
                confidence_column
            ] = "Classifier's confidence in the category {}".format(str(n + 1))

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        categories = sorted(
            response.get("categories", []),
            key=lambda x: x.get("confidence"),
            reverse=True,
        )
        for n in range(self.num_categories):
            category_column = generate_unique(
                "category_" + str(n + 1) + "_name", row.keys(), self.column_prefix
            )
            confidence_column = generate_unique(
                "category_" + str(n + 1) + "_confidence", row.keys(), self.column_prefix
            )
            if len(categories) > n:
                row[category_column] = categories[n].get("name", "")
                row[confidence_column] = categories[n].get("confidence")
            else:
                row[category_column] = ""
                row[confidence_column] = None
        return row
