drop view if exists property_values;

create view property_values as (
    select
        id
        , pid
        , emv_total as value_
        , valid
    from
        parcel);

