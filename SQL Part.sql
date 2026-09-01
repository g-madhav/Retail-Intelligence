USE retail_db;

SELECT COUNT(*) AS total_rows
FROM sales_data;

DESCRIBE sales_data;

SELECT *
FROM sales_data
LIMIT 10;

SELECT
product_category_name,
SUM(qty) AS total_sales
FROM sales_data
GROUP BY product_category_name
ORDER BY total_sales DESC
LIMIT 10;

SELECT
product_category_name,
SUM(total_price) AS revenue
FROM sales_data
GROUP BY product_category_name
ORDER BY revenue DESC
LIMIT 10;

SELECT
holiday,
AVG(qty) AS avg_sales
FROM sales_data
GROUP BY holiday;

SELECT
weekend,
AVG(qty) AS avg_sales
FROM sales_data
GROUP BY weekend;

SELECT
month,
SUM(qty) AS sales
FROM sales_data
GROUP BY month
ORDER BY month;
