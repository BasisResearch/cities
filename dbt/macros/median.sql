{% macro median(attr) %}
(percentile_cont(0.5) within group (order by {{ attr }}))
{% endmacro %}
