SELECT 
    *
FROM TICKER
where opentime >= START_MILLI
order by opentime;