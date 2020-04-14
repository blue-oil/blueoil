Converter Optimization passes (tentative, only for v0.2.0)
=======================================
| author: Takeo Imai
| date: 2018.4.25

| This is the document for checking the current status of optimization structure in Converter.
| The structure would be changed/restructured, so this document would be deleted accordingly.

Current use of opimization passes
-----------------------
.. highlight:: python

Currently, the optimization passes are implemented as methods of a class.
And these methods are called in ``blueoil.converter.generate_project#optimize_graph_step``

.. literalinclude:: ../../blueoil/converter/generate_project.py
   :language: python
   :lines: 59-68

| First of all, this removes `axis` node from `concat` node.
| Then, if the hard quantization is active:

#. firstly transpose the weight of `convolution` node from *HWCN* to *NHWC*, then
#. precompute the weight data of `convolution` nodes beforehand, which is mainly for packing multiple quantized float weight into a integer word. And finally,
#. change the type of nodes between `convolution` and `quantize` nodes. Hard quantization changes the data type from float to integer.

| Otherwise,

#. transpose the weight of **quantized** convolution nodes from ‘HWCN’ to ‘NHWC’, then
#. (this is a temporal implementation but) transpose the weight of **non-quantized** convolution nodes like the above.

| The other final optimization is *threshold skipping*, which prunes all nodes between `convolution` and its quantization, so it can lead to less computational time.


Optimizer class
------------------
Class ``blueoil.converter.core.optimizer`` is the current implementation of the optimization passes.
These passes are implemented as twelves methods in the class as the following.

.. automodule:: blueoil.converter.core.optimizer
   :members:
   :noindex:
..   :undoc-members:
   :show-inheritance:
