{% macro standardize_cont(columns) %}
 {% for c in columns %}
   {{ c }} as {{ c }}_original, (({{ c }} - (avg({{ c }}) over ())) / (stddev_samp({{ c }}) over ()))::double precision as {{ c }}
   {% if not loop.last %},{% endif %}
 {% endfor %}
{% endmacro %}

{% macro standardize_cat(columns) %}
 {% for c in columns %}
   {{ c }} as {{ c }}_original, (dense_rank() over (order by {{ c }})) - 1 as {{ c }}
   {% if not loop.last %},{% endif %}
 {% endfor %}
{% endmacro %}
