{% set years = range(2018, 2024) %}

{% for year_ in years %}
  select
    "YYYYMM" as yyyy_mm
    , "ZIPCODE" as zip_code
    , "CITY" as city
    , "STATE" as state_
    , "TOTAL_FROM_ZIP" as total_from_zip
    , "TOTAL_BUSINESS" as total_from_zip_business
    , "TOTAL_FAMILY" as total_from_zip_family
    , "TOTAL_INDIVIDUAL" as total_from_zip_individual
    , "TOTAL_PERM" as total_from_zip_perm
    , "TOTAL_TEMP" as total_from_zip_temp
    , "TOTAL_TO_ZIP" as total_to_zip
    , "TOTAL_BUSINESS_dup" as total_to_zip_business
    , "TOTAL_FAMILY_dup" as total_to_zip_family
    , "TOTAL_INDIVIDUAL_dup" as total_to_zip_individual
    , "TOTAL_PERM_dup" as total_to_zip_perm
    , "TOTAL_TEMP_dup" as total_to_zip_temp
  from {{ source('minneapolis', 'usps_y' ~ year_) }}
{% if not loop.last %} union all {% endif %}
{% endfor %}
