All content is from taken from the IBM Bluemix Watson Developer Cloud Documentation.

# Overview
IBM Watson™ Discovery makes it possible to rapidly build cognitive, cloud-based exploration applications that unlock actionable insights hidden in unstructured data — including your own proprietary data, as well as public and third-party data.

**Discovery is aviailable in English, Spanish and German.**

### Quick notes
* Max file size: 50MB.
* Accepts the following formats: HTML, JSON, DOC, PDF.
* Languages cannot be mixed in the same collection.
* Adding a different config file to a collection will not change the residing data on that collection.
* Text enrichments include: Sentiment analysis, concept tagging, keywords, semantic roles, category classification, emotion analysis and entity extraction.

### Adding Methods:
* API: Use if you are integrating the upload content within an existing application or creating a custom upload.
* Tooling: Normal usage
* Data Crawler: Use of you want to manage a large file upload or want to extract content from different repositories (ex. a database).

### Steps to to use the Discovery Service:
1. Create an environment with either a default or custom configuration. A default configuration will enrich your data with the Natural Language Understnading Service.
2. If necessary, create different collections to place your data. A common practice is to create a test collection with test data and a another with production-ready data.
3. Upload your documents to your collection.
4. If you selected default configuration, you can select what kind of enrichments your text will create.

Note: Some reasons for having multiple collections include:
* Seperating results for different audiences.
* Data may be so different that it doesn't make sense to query all the data at the same time.




