# MapIntel Data

This is the Data collection repository of the MapIntel project. This repository contains all the data collection and pre-processing pipelines used to feed the MapIntel system.

The MapIntel project repository, containing its codebase and instructions on how to use it, can be found at [github.com/NOVA-IMS-Innovation-and-Analytics-Lab/mapintel_project](https://github.com/NOVA-IMS-Innovation-and-Analytics-Lab/mapintel_project).

## Usage
For the newsapi AWS SAM data collection app find more information in the README file inside the corresponding folder.
For the preprocessing pipelines inside the preprocess folder, you just need to execute the scripts for the data that is needed.

## The data structure
Every preprocess script should output a json file with the following key-value pairs:
- *text*: text that the model will use to represent the document;
- *title*: title used by the user interface to represent the document;
- *url*: url used to access the full original document from its source;
- *timestamp*: datetime string that marks the release of the original document. Formatted as "%d-%m-%Y %H:%M:%S";
- *snippet*: **optional** excerpt of the document displayed in the user interface. If not given, text will be used for this function;
- *image_url*: **optional** headline image url of the document used in the user interface. If not given, a placeholder image is used instead;

Example of a single document representation:
``` python
{
  'text': 'Eagles to start Jalen Mills at cornerback, Marcus Epps at safety against 49ers | Report - Bleeding Green Nation Secondary change up. The Philadelphia Eagles will be starting Jalen Mills at cornerback and Marcus Epps at safety in their Week 4 game against the San Francisco 49ers, according to one report:#Eagles lineup changes, pe',
  'image_url': 'https://cdn.vox-cdn.com/thumbor/zAtOYRtDGrSfDrlfk1gh2VHHAjQ=/0x167:1883x1153/fit-in/1200x630/cdn.vox-cdn.com/uploads/chorus_asset/file/20098871/usa_today_13755700.jpg',
  'url': 'https://www.bleedinggreennation.com/2020/10/4/21501492/jalen-mills-eagles-vs-49ers-cornerback-safety-marcus-epps-kvon-wallace-philadelphia-nfl-news-game',
  'title': 'Eagles to start Jalen Mills at cornerback, Marcus Epps at safety against 49ers | Report - Bleeding Green Nation',
  'snippet': 'Secondary change up. The Philadelphia Eagles will be starting Jalen Mills at cornerback and Marcus Epps at safety in their Week 4 game against the San Francisco 49ers, according to one report:#Eagles lineup changes, pe',
  'timestamp': '04-10-2020 21:52:27'
}
```

## Project Organization

    ├── collection
    │   │
    │   └── newsapi_sam_app         <- AWS SAM App used to retrieve news articles from NewsAPI on a schedule
    │
    ├── preprocess
    │   │
    │   ├── arquivo.py              <- Arquivo.pt data preprocessing pipeline
    │   └── newsapi.py              <- NewsAPI data preprocessing pipeline
    │
    └── README.md

