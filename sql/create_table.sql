CREATE TABLE IF NOT EXISTS TICKER_TABLE_NAME (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `opentime` TIMESTAMP null,
    `open` DOUBLE,
    `high` DOUBLE,
    `low` DOUBLE,
    `close` DOUBLE,
    `volume` DOUBLE,
    `closetime` TIMESTAMP null,
    `quote_asset_volume` DOUBLE,
    `num_of_trades` BIGINT,
    `taker_by_base` DOUBLE,
    `taker_buy_quote` DOUBLE,
    `ignore` VARCHAR(10)
);

ALTER TABLE TICKER_TABLE_NAME ADD UNIQUE INDEX(`opentime`);