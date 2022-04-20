SELECT 
    *
FROM TICKER
where opentime >= START_MILLI and closetime <= END_MILLI
order by opentime;