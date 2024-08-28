{% macro standardize(columns) %}
 {% for c in columns %}
   {{ c }} as {{ c }}_original, (({{ c }} - (avg({{ c }}) over ())) / (stddev_samp({{ c }}) over ()))::double precision as {{ c }}
   {% if not loop.last %},{% endif %}
 {% endfor %}
{% endmacro %}
