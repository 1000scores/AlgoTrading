select 
    a.*
from (
    SELECT
        *
    FROM TICKER
    order by opentime
    limit N_OHLCV
) a
order by a.opentime asc