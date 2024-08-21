{% macro standardize(columns) %}
 {% for c in columns %}
   (({{ c }} - (avg({{ c }}) over ())) / (stddev_samp({{ c }}) over ())) as std_{{ c }}
   {% if not loop.last %},{% endif %}
 {% endfor %}
{% endmacro %}
