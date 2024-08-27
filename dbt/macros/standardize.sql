{% macro standardize(columns) %}
 {% for c in columns %}
   (({{ c }} - (avg({{ c }}) over ())) / (stddev_samp({{ c }}) over ()))::double precision as std_{{ c }}
   {% if not loop.last %},{% endif %}
 {% endfor %}
{% endmacro %}
