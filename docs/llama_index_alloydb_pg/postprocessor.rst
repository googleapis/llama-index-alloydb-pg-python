Node Postprocessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AlloyDBRerank`` is a LlamaIndex node postprocessor that reranks retrieved nodes in the database with AlloyDB AI's ``google_ml.rank`` function. ``model_id`` is required.

.. note::
   ``google_ml.rank`` needs the ``google_ml_integration.enable_ai_query_engine`` database flag turned on, and the AlloyDB service agent needs the ``roles/discoveryengine.viewer`` IAM role to call the Vertex AI ranking API.

.. code-block:: python

    from llama_index_alloydb_pg import AlloyDBRerank

    reranker = AlloyDBRerank(
        engine=engine, model_id="semantic-ranker-default-003", top_n=3
    )
    query_engine = index.as_query_engine(
        similarity_top_k=10, node_postprocessors=[reranker]
    )

.. automodule:: llama_index_alloydb_pg.postprocessor
  :members:
  :private-members:
  :noindex:
