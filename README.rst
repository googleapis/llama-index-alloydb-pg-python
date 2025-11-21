AlloyDB for PostgreSQL for LlamaIndex
==================================================

|preview| |pypi| |versions|

- `Client Library Documentation`_
- `Product Documentation`_

The **AlloyDB for PostgreSQL for LlamaIndex** package provides a first class experience for connecting to
AlloyDB instances from the LlamaIndex ecosystem while providing the following benefits:

- **Simplified & Secure Connections**: easily and securely create shared connection pools to connect to Google Cloud databases utilizing IAM for authorization and database authentication without needing to manage SSL certificates, configure firewall rules, or enable authorized networks.
- **Better integration with AlloyDB**: built-in methods to take advantage of AlloyDB's advanced indexing and scalability capabilities.
- **Improved metadata handling**: store metadata in columns instead of JSON, resulting in significant performance improvements.
- **Clear separation**: clearly separate table and extension creation, allowing for distinct permissions and streamlined workflows.

.. |preview| image:: https://img.shields.io/badge/support-preview-orange.svg
   :target: https://github.com/googleapis/google-cloud-python/blob/main/README.rst#stability-levels
.. |pypi| image:: https://img.shields.io/pypi/v/llama-index-alloydb-pg.svg
   :target: https://pypi.org/project/llama-index-alloydb-pg/
.. |versions| image:: https://img.shields.io/pypi/pyversions/llama-index-alloydb-pg.svg
   :target: https://pypi.org/project/llama-index-alloydb-pg/
.. _Client Library Documentation: https://cloud.google.com/python/docs/reference/llama-index-alloydb-pg/latest
.. _Product Documentation: https://cloud.google.com/alloydb

Quick Start
-----------

In order to use this library, you first need to go through the following
steps:

1. `Select or create a Cloud Platform project.`_
2. `Enable billing for your project.`_
3. `Enable the AlloyDB API.`_
4. `Setup Authentication.`_

.. _Select or create a Cloud Platform project.: https://console.cloud.google.com/project
.. _Enable billing for your project.: https://cloud.google.com/billing/docs/how-to/modify-project#enable_billing_for_a_project
.. _Enable the AlloyDB API.: https://console.cloud.google.com/flows/enableapi?apiid=alloydb.googleapis.com
.. _Setup Authentication.: https://googleapis.dev/python/google-api-core/latest/auth.html

Installation
~~~~~~~~~~~~

Install this library in a `virtualenv`_ using pip. `virtualenv`_ is a tool to create isolated Python environments. The basic problem it addresses is
one of dependencies and versions, and indirectly permissions.

With `virtualenv`_, it's
possible to install this library without needing system install
permissions, and without clashing with the installed system
dependencies.

.. _`virtualenv`: https://virtualenv.pypa.io/en/latest/

Supported Python Versions
^^^^^^^^^^^^^^^^^^^^^^^^^

Python >= 3.10

Mac/Linux
^^^^^^^^^

.. code-block:: console

   pip install virtualenv
   virtualenv <your-env>
   source <your-env>/bin/activate
   <your-env>/bin/pip install llama-index-alloydb-pg

Windows
^^^^^^^

.. code-block:: console

    pip install virtualenv
    virtualenv <your-env>
    <your-env>\Scripts\activate
    <your-env>\Scripts\pip.exe install llama-index-alloydb-pg

Example Usage
-------------

Code samples and snippets live in the `samples/`_ folder.

.. _samples/: https://github.com/googleapis/llama-index-alloydb-pg-python/tree/main/samples

Vector Store Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a vector store to store embedded data and perform vector search.

.. code-block:: python

   import google.auth
   from llama_index.core import Settings
   from llama_index.embeddings.vertex import VertexTextEmbedding
   from llama_index_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore


   credentials, project_id = google.auth.default()
   engine = await AlloyDBEngine.afrom_instance(
      "project-id", "region", "my-cluster", "my-instance", "my-database"
   )
   Settings.embed_model = VertexTextEmbedding(
      model_name="textembedding-gecko@003",
      project="project-id",
      credentials=credentials,
   )

   vector_store = await AlloyDBVectorStore.create(
      engine=engine, table_name="vector_store"
   )


Chat Store Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~

A chat store serves as a centralized interface to store your chat history.

.. code-block:: python

   from llama_index.core.memory import ChatMemoryBuffer
   from llama_index_cloud_sql_pg import AlloyDBChatStore, AlloyDBEngine


   engine = await AlloyDBEngine.afrom_instance(
      "project-id", "region", "my-cluster", "my-instance", "my-database"
   )
   chat_store = await AlloyDBChatStore.create(
      engine=engine, table_name="chat_store"
   )
   memory = ChatMemoryBuffer.from_defaults(
      token_limit=3000,
      chat_store=chat_store,
      chat_store_key="user1",
   )


Document Reader Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~

A Reader ingest data from different data sources and data formats into a simple `Document` representation.

.. code-block:: python

   from llama_index.core.memory import ChatMemoryBuffer
   from llama_index_cloud_sql_pg import AlloyDBReader, AlloyDBEngine


   engine = await AlloyDBEngine.afrom_instance(
      "project-id", "region", "my-cluster", "my-instance", "my-database"
   )
   reader = await AlloyDBReader.create(
      engine=engine, table_name="my-db-table"
   )
   documents = reader.load_data()


Document Store Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a document store to make storage and maintenance of data easier.

.. code-block:: python

   from llama_index_alloydb_pg import AlloyDBEngine, AlloyDBDocumentStore


   engine = await AlloyDBEngine.afrom_instance(
      "project-id", "region", "my-cluster", "my-instance", "my-database"
   )
   doc_store = await AlloyDBDocumentStore.create(
      engine=engine, table_name="doc_store"
   )


Index Store Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use an index store to keep track of indexes built on documents.

.. code:: python

   from llama_index_alloydb_pg import AlloyDBIndexStore, AlloyDBEngine


   engine = await AlloyDBEngine.from_instance(
      "project-id", "region", "my-cluster", "my-instance", "my-database"
   )
   index_store = await AlloyDBIndexStore.create(
      engine=engine, table_name="index_store"
   )


Contributions
~~~~~~~~~~~~~

Contributions to this library are always welcome and highly encouraged.

See `CONTRIBUTING`_ for more information how to get started.

Please note that this project is released with a Contributor Code of Conduct. By participating in
this project you agree to abide by its terms. See `Code of Conduct`_ for more
information.

.. _`CONTRIBUTING`: https://github.com/googleapis/llama-index-alloydb-pg-python/tree/main/CONTRIBUTING.md
.. _`Code of Conduct`: https://github.com/googleapis/llama-index-alloydb-pg-python/tree/main/CODE_OF_CONDUCT.md

License
-------

Apache 2.0 - See
`LICENSE <https://github.com/googleapis/llama-index-alloydb-pg-python/tree/main/LICENSE>`_
for more information.

Disclaimer
----------

This is not an officially supported Google product.
