drop table if exists fair_market_rents_raw cascade;

create table fair_market_rents_raw (
    zip text
    , rent_br0 numeric
    , rent_br1 numeric
    , rent_br2 numeric
    , rent_br3 numeric
    , rent_br4 numeric
    , year_ int
);

