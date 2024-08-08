drop table if exists fair_market_rents cascade;

create table fair_market_rents (
    zip_id int references zip_code (id)
    , rent numeric
    , num_bedrooms int
    , year_ int
);

insert into fair_market_rents (zip_id , rent , num_bedrooms , year_)
with fmr_zip as (
    select
        zip_code.id as zip_id
        , rent_br0
        , rent_br1
        , rent_br2
        , rent_br3
        , rent_br4
        , year_
    from
        fair_market_rents_raw
        join zip_code on zip_code.zip_code = fair_market_rents_raw.zip
            and zip_code.valid @> to_date(year_::text , 'YYYY'))
    select
        zip_id
        , rent_br0
        , 0
        , year_
    from
        fmr_zip
    union
    select
        zip_id
        , rent_br1
        , 1
        , year_
    from
        fmr_zip
    union
    select
        zip_id
        , rent_br2
        , 2
        , year_
    from
        fmr_zip
    union
    select
        zip_id
        , rent_br3
        , 3
        , year_
    from
        fmr_zip
    union
    select
        zip_id
        , rent_br4
        , 4
        , year_
    from
        fmr_zip;

