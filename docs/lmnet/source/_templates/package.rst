{{ fullname }} package
{% for item in range(8 + fullname|length) -%}={%- endfor %}

.. automodule:: {{ fullname }}
    {% if members -%}
    :members: {{ members|join(", ") }}
    :undoc-members:
    :show-inheritance:
    {%- endif %}

{% if submodules %}
    Submodules:

    .. toctree::
       :maxdepth: 1
{% for item in submodules %}
       {{ fullname }}.{{ item }}
       {%- endfor %}
    {%- endif -%}

{% if subpackages %}
    Subpackages:

    .. toctree::
       :maxdepth: 1
{% for item in subpackages %}
       {{ fullname }}.{{ item }}
       {%- endfor %}
    {%- endif %}

{% if members %}
    Summary
    -------

    {%- if exceptions %}

    Exceptions:

    .. autosummary::
        :nosignatures:
{% for item in exceptions %}
        {{ item }}
{%- endfor %}
    {%- endif %}

    {%- if classes %}

    Classes:

    .. autosummary::
        :nosignatures:
{% for item in classes %}
        {{ item }}
{%- endfor %}
    {%- endif %}

    {%- if functions %}

    Functions:

    .. autosummary::
        :nosignatures:
{% for item in functions %}
        {{ item }}
{%- endfor %}
    {%- endif %}
{%- endif %}

    {%- if data %}

    Data:

    .. autosummary::
        :nosignatures:
{% for item in data %}
        {{ item }}
{%- endfor %}
    {%- endif %}

{% if all_refs %}
    ``__all__``: {{ all_refs|join(", ") }}
{%- endif %}


{% if members %}
    Reference
    ---------

{%- endif %}
