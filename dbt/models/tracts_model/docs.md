{% docs tracts_model_int__census_tracts_filtered %}

Intermediate table that selects census tracts of interest. Considers only tracts
in the city boundary (tracts must intersect boundary and have at least 90% of
area overlapping) and only for years 2011 to 2020.

Notes:
- Census tracts for 2020 are replaced with tracts for 2019. This requires
  retagging parcels and other spatial entities, because the `census_tract_id`
  changes with the replacement.

{% enddocs %}

{% docs tracts_model_int__parcels_filtered %}

Retag parcels to account for tract replacement. This also has the effect of
filtering parcels to the considered tracts.

{% enddocs %}

{% docs census_tracts_distance_to_transit %}

Aggregate `parcels_distance_to_transit` by tract.

{% enddocs %}

{% docs census_tracts_housing_units %}

Aggregate number of units built by tract. Unit data is drawn from
`residential_permits`.

{% enddocs %}

{% docs census_tracts_parcel_area %}

Aggregate parcel area by tract. Area is computed from the parcel geometry, not
from the area included in the parcel dataset.

{% enddocs %}

{% docs census_tracts_parking_limits %}

Parking limits aggregated by tract.

{% enddocs %}

{% docs parcels_distance_to_transit %}

Distance from a parcel to the nearest transit (line or stop). This is the
smallest distance from the parcel geometry to the line geometry, not from the
parcel centroid.

{% enddocs %}

{% docs parcels_parking_limits %}

Parking limits by parcel. The parking limit is a function of the distance from
the parcel to the nearest transit line/transit stop.

Notes:
- Parcels in all years that intersect (any level of intersection) the downtown
  area have the limit eliminated.
- Parcels before 2015 have the full limit.
- Parcels after 2015 and in the blue zone have the limit eliminated.
- Parcels after 2015 and in the yellow zone have the limit reduced.

{% enddocs %}

{% docs census_tracts_property_values %}

Total and median property value aggregated by tract. Uses total estimated market
value from the parcel dataset.

{% enddocs %}

{% docs tracts_model__census_tracts %}

Wide table that joins various census tract level aggregates.

Notes:
- Continuous columns are standardized by default. Categorical columns are
  remapped to [0, |D|), where D is the domain. The original value of a column
  `c` is called `c_original`.
- Demographic variables are drawn from ACS tract level data.

{% enddocs %}

{% docs tracts_model__parcels %}

Parcels filtered by the considered census tracts, with additional data.

{% enddocs %}
