{% macro safe_divide(num, dem) %}
  (case when {{ dem }} = 0 then 0 else {{ num }} / {{ dem }} end)
{% endmacro %}
