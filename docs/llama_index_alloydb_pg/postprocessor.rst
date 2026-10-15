Node Postprocessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AlloyDBRerank`` is a LlamaIndex node postprocessor that reranks retrieved nodes in the database with AlloyDB AI's ``google_ml.rank`` function. ``model_id`` is required.

.. note::
   Prerequisites (see `Rank and rerank search results <https://cloud.google.com/alloydb/docs/ai/rank-rerank-search-results-rag>`_): the ``google_ml_integration`` extension 1.5.2 or later, the ``google_ml_integration.enable_model_support`` and ``google_ml_integration.enable_ai_query_engine`` database flags turned on, the Discovery Engine API (``discoveryengine.googleapis.com``) enabled, and the ``roles/discoveryengine.viewer`` IAM role for the AlloyDB service agent.

.. code-block:: python

    from llama_index_alloydb_pg import AlloyDBRerank

    reranker = AlloyDBRerank(engine=engine, model_id="semantic-ranker-default-003", top_n=3)
    query_engine = index.as_query_engine(similarity_top_k=10, node_postprocessors=[reranker])

.. automodule:: llama_index_alloydb_pg.postprocessor
  :members:
  :private-members:
  :noindex:
