"""Simple script to communicate with arquivo.pt API."""
from os.path import join
import logging
import time
import multiprocessing.dummy as mp
from multiprocessing import cpu_count
from itertools import product
from sqlite3 import connect
import requests
import pandas as pd
from bs4 import BeautifulSoup


class ArquivoPT:
    """
    Query Arquivo.pt using search terms.
    EndPoint: https://arquivo.pt/textsearch

    The URL search feature is also available using the ``website``
    parameter.

    Note: a text search query returns a maximum of 2000 response results

    More info is available here:
    https://github.com/arquivo/pwa-technologies/wiki/Arquivo.pt-API#request-parameters

    Parameters
    ----------
    query : str (default=None)
        Query search terms.
        The query can contain advanced search operators. Advanced search
        operators can be:
            * " " : search for items containing expression composed by search
                terms (e.g. phrase or named entity).
            * - : excludes items that contains the given terms.

    website : str (default=None)
        The only parameter required for URL search. If not None, the remaining
        parameters will be ignored.

        It will return a list of the preserved versions for the URL that was
        required. The URL may or may not contain the protocol, eg. http. Being
        strongly advised to define the url with the respective protocol. You
        must encode the originalURL with the percent-encoding (URL encoding).

    from_date : str (default="1996")
        Set an initial date for the time span of the search.
        Format: YYYYMMDDHHMMSS, also accepts a shorter date fotmat,
        e.g., (YYYY).

    to_date : str (default=None)
        Set a end date for the time span of the search.
        Format: YYYYMMDDHHMMSS, also accepts a shorter date format, for
        example (YYYY). If None, defaults to Current Year-1

    formats : str (default=None)
        Specify accepted formats for the response items.
        Subtype of the MIME types (e.g. pdf, ps, html, xls, ppt, doc, rtf,
        etc).

    offset : int (default=0)
        The position of the text indices where the search begins.

    site_search : str (default=None)
        Limit search within a given site.

    collection : str (default=None)
        Limit search within a given collection. Only results from the
        specified collections are return. A list of all the collections
        preserved by Arquivo.pt is publicly available in the link provided
        in the description of this class.

    max_items : int (default=50)
        Maximum number of items on the response. Max: 2000

    dedup_value : int (default=2)
        Maximum number of items per dedupField.

    dedup_field : str (default="site")
        Result field where the deduplication will be performed. (ex. site, url)

    fields : str (default=None)
        Selector specifying a subset of fields to include in the response.
        Separated by ",". Possible fields: title, originalURL, linkToArchive,
        tstamp, contentLength, digest, mimeType, linkToScreenshot, date,
        encoding, linkToNoFrame, linkToOriginalFile, collection, snippet,
        linkToExtractedText

    callback : str (default=None)
        Callback function. For more information, see the partial request
        section in the REST from JavaScript (see URL provided above). Use for
        better performance.

    pretty_print : bool (default=True)
        Returns response with indentations and line breaks.
        - Returns the response in a human-readable format if true.
        - Default value: true.
        - When this is false, it can reduce the response payload size, which
          might lead to better performance in some environments.


    """

    def __init__(
        self,
        query=None,
        website=None,
        from_date=None,
        to_date=None,
        formats=None,
        offset=None,
        site_search=None,
        collection=None,
        max_items=None,
        dedup_value=None,
        dedup_field=None,
        fields=None,
        callback=None,
        pretty_print=True,
    ):
        self.query = query
        self.website = website
        self.from_date = from_date
        self.to_date = to_date
        self.formats = formats
        self.offset = offset
        self.site_search = site_search
        self.collection = collection
        self.max_items = max_items
        self.dedup_value = dedup_value
        self.dedup_field = dedup_field
        self.fields = fields
        self.callback = callback
        self.pretty_print = pretty_print

    def _get_request_params(self):
        """Generates a dictionary with the query parameters."""

        pretty_print = (
            str(self.pretty_print).lower() if self.pretty_print is not None else None
        )

        params = {
            "from": self.from_date,
            "to": self.to_date,
            "type": self.formats,
            "offset": self.offset,
            "siteSearch": self.site_search,
            "collection": self.collection,
            "maxItems": self.max_items,
            "dedupValue": self.dedup_value,
            "dedupField": self.dedup_field,
            "fields": self.fields,
            "callback": self.callback,
            "prettyPrint": pretty_print,
        }

        if self.website is not None:
            params["versionHistory"] = self.website
        else:
            params["q"] = self.query

        return {k: v for k, v in params.items() if v is not None}

    def _remove_html_tags(self, text):
        return " ".join(BeautifulSoup(text, "html.parser").stripped_strings)

    def data_pull(self):
        """
        Sends a GET request to the Arquivo.pt API and stores content as a dict.
        """
        endpoint = "https://arquivo.pt/textsearch"
        params = self._get_request_params()

        # Get data (+ raise error if it fails to do so)
        self.response_ = requests.get(url=endpoint, params=params)
        self.response_.raise_for_status()

        # Store data
        self.content_ = self.response_.json()
        return self

    def to_dataframe(self):
        """Preprocesses and moves data into a pandas dataframe."""
        df = pd.DataFrame(self.content_["response_items"])
        if self.website is None:
            text_cols = ["title", "snippet"]
            for col in text_cols:
                df[col] = df[col].apply(self._remove_html_tags)
        return df


class ArquivoPages(ArquivoPT):
    """
    Pull data from specific websites in Arquivo.pt.

    This class performs auto-pagination, error handling, and breaks down data pulls by
    year/month/day to pull as much data from Arquivo as possible.

    Parameters
    ----------
    site_search : str, list (default=None)
        Limit search within a given site or list of sites.

    from_date : str (default="1996")
        Set an initial date for the time span of the search.
        Format: YYYYMMDDHHMMSS, also accepts a shorter date fotmat,
        e.g., (YYYY).

    to_date : str (default=None)
        Set a end date for the time span of the search.
        Format: YYYYMMDDHHMMSS, also accepts a shorter date format, for
        example (YYYY). If None, defaults to Current Year-1

    date_chunks : int (default=None)
        Number of chunks to split the date range into. If None, uses a single chunk.

    max_items : int (default=50)
        Maximum number of items on the response. Max: 2000

    n_jobs : int, optional (default=-1)
        The number of threads to open if possible. A value of -1 uses all available
        threads.

    verbose : int (default=0)
        Controls the verbosity: the higher, the more messages.

        - >=1 : Displays api request limit warnings and invalid requests.
        - >=2 : Displays all the requests being performed.
    """
    def __init__(
        self,
        site_search=None,
        from_date=None,
        to_date=None,
        date_chunks=None,
        max_items=2000,
        n_jobs=-1,
        verbose=0
    ):
        self.site_search = site_search
        self.from_date = from_date
        self.to_date = to_date
        self.date_chunks = date_chunks
        self.max_items = max_items
        self.n_jobs = n_jobs

        if verbose >= 2:
            logging.basicConfig(
                level=logging.INFO,
                format='[%(asctime)s] %(levelname)s - %(message)s'
            )
        elif verbose >= 1:
            logging.basicConfig(
                level=logging.WARNING,
                format='[%(asctime)s] %(levelname)s - %(message)s'
            )

    def _get_request_params(self, website, from_date, to_date, offset):
        """Generates a dictionary with the query parameters."""

        params = {
            "q": "",
            "from": from_date,
            "to": to_date,
            "offset": offset,
            "siteSearch": website,
            "maxItems": self.max_items,
        }

        return {k: v for k, v in params.items() if v is not None}

    def _single_data_pull(self, website, from_date, to_date, offset):
        """
        Sends a GET request to the Arquivo.pt API and stores content as a dict.
        """
        endpoint = "https://arquivo.pt/textsearch"
        params = self._get_request_params(website, from_date, to_date, offset)

        # Get data (+ raise error if it fails to do so)
        response_ = requests.get(url=endpoint, params=params)
        if response_.status_code == 429:
            # If the maximum number of requests have been hit
            logging.warning(
                "Max requests hit. Continuing data pull in 1 min. " +
                f"[website: {website}, from_date:{from_date}, to_date: {to_date}, " +
                f"offset: {offset}]"
            )
            time.sleep(61)
            return self._single_data_pull(website, from_date, to_date, offset)
        elif response_.status_code == 500:
            # The request made is not possible to get
            logging.warning(
                "500'ed. No items available for this request. " +
                f"[website: {website}, from_date:{from_date}, to_date: {to_date}, " +
                f"offset: {offset}]"
            )
            return None
        else:
            response_.raise_for_status()
            response = response_.json()

            logging.info(
                f"[website: {website}, from_date:{from_date}, to_date: {to_date}, " +
                f"offset: {offset}] Nbr existing: {response['estimated_nr_results']} " +
                f"| Nbr pulled: {len(response['response_items'])}"
            )

            return response

    def _to_dataframe(self, responses):
        """Preprocesses and moves data into a pandas dataframe."""

        response_items = [
            response["response_items"] for response in responses
            if response is not None
            and "response_items" in response.keys()
            and len(response["response_items"])
        ]

        responses_dfs = []
        for response in response_items:
            df = pd.DataFrame(response)
            text_cols = ["title", "snippet"]
            for col in text_cols:
                df[col] = df[col].apply(self._remove_html_tags)
            responses_dfs.append(df)

        df_all = (
            pd.concat(responses_dfs).drop_duplicates()
            if len(responses_dfs) > 0
            else pd.DataFrame()
        )
        return df_all

    def check_params(self):

        date_formatter = (
            lambda date: str(date)
            .replace(" ", "")
            .replace("-", "")
            .replace(":", "")
            .split(".")[0]
        )
        # break down date range into smaller chunks
        dates = pd.date_range(
            start=self.from_date, end=self.to_date, periods=self.date_chunks
        )
        date_limits = [
            (date_formatter(dates[i]), date_formatter(dates[i+1]))
            for i in range(len(dates)-1)
        ]

        # set up offset range
        offsets = range(0, 2000+self.max_items, self.max_items)

        # Set up iterables
        iterables = [
            (from_date, to_date, offset)
            for ((from_date, to_date), offset)
            in product(date_limits, offsets)
        ]

        # check if website is a list of strings
        websites = (
            self.site_search if type(self.site_search) == list else [self.site_search]
        )

        return websites, iterables

    def data_pull(self):

        # set up data pull parameters
        websites, iterables = self.check_params()
        n_jobs = self.n_jobs if self.n_jobs != -1 else cpu_count()

        p = mp.Pool(n_jobs)

        # loop by website, date chunk, offset
        self.content_ = []
        for website in websites:
            responses = p.map(
                lambda params: self._single_data_pull(website, *params), iterables
            )
            self.content_.append((website, self._to_dataframe(responses)))

        return self

    def save(self, path, db_name):
        """Save datasets."""
        with connect(join(path, f"{db_name}.db")) as connection:
            for name, data in self.content_:
                if data.shape[0] > 0:
                    data.to_sql(name, connection, index=False, if_exists="replace")
