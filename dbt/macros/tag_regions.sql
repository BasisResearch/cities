-- Tag regions with their containing/most intersecting/closest parent regions.
-- child_table: table with the child regions
-- parent_table: table with the parent regions
-- max_distance: maximum distance to consider a region as a parent (meters)
{% macro tag_regions(child_table, parent_table, max_distance=100) %}
(
with child as (
    select * from {{child_table}}
)
, parent as (
    select * from {{parent_table}}
)
, within as (
    select child.id as child_id
        , parent.id as parent_id
        , child.valid * parent.valid as valid
    from
        child
        inner join parent
            on ST_Within (child.geom, parent.geom)
            and child.valid && parent.valid
)
, not_within as (
    select * from child
    where not exists (select child_id from within where child_id = id)
)
, largest_overlap as (
    select distinct on (child.id)
        child.id as child_id
        , parent.id as parent_id
        , child.valid * parent.valid as valid
    from
        not_within as child
        inner join parent
            on ST_Intersects (child.geom, parent.geom)
            and child.valid && parent.valid
    order by
      child_id,
      ST_Area (ST_Intersection (child.geom, parent.geom)) desc
)
, no_overlap as (
    select * from not_within
    where not exists (
      select child_id from largest_overlap where child_id = id
    )
)
, closest as (
    select distinct on (child.id)
        child.id as child_id
        , parent.id as parent_id
        , child.valid * parent.valid as valid
    from
        no_overlap as child
        inner join parent
            on child.valid && parent.valid
            and ST_DWithin (child.geom, parent.geom, {{max_distance}})
    order by
      child_id,
      ST_Distance (child.geom, parent.geom)
)
select *, 'within' as type_ from within
union all
select *, 'most_overlap' as type_ from largest_overlap
union all
select *, 'closest' as type_ from closest
)
{% endmacro %}
