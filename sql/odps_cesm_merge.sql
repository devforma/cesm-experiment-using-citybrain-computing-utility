---------------------预处理---------------------

----将列数据的格式改成指定格式
--ALTER TABLE dbo.temp1
--ALTER COLUMN lon float


---------------------超参数---------------------
----定义event标准
--分位数要求定义
--DECLARE @quant float = 0.9;
--持续事件要求定义
--DECLARE @duration_threshold int = 3;
---------------------输入数据---------------------
--DECLARE @var_col_name varchar(20) = 'climate_index';
--DECLARE @lat_col_name varchar(20) = 'lat';
--DECLARE @lon_col_name varchar(20) = 'lon';
--DECLARE @time_col_name varchar(20) = 'timelist';
--DECLARE @start_time varchar(20) = 'start_time';
--DECLARE @end_time varchar(20) = 'end_time';
--DECLARE @member_id int = 1;
--DECLARE @SELECT_ALL_TIME BOOLEAN = TRUE;
--DECLARE @output_table varchar(20) = 'output_table';
--DECLARE @set_quantile FLOAT = 35;
---------------------基础表---------------------

--定义T0为对某个经纬度与时间（天为单位下）的所有数据取平均
WITH T0 AS (
SELECT 
	@var_col_name climate_index,
	@lat_col_name lat,
	@lon_col_name lon,
	@time_col_name time_list
	
FROM

	@tablename 

WHERE 
	(
			(
			@SELECT_ALL_TIME = 1 OR 
				(
			YEAR(@time_col_name) >= @start_year AND
			YEAR(@time_col_name) <= @end_year AND
			MONTH(@time_col_name) >= @start_month AND
			MONTH(@time_col_name) <= @end_month
				)
			)
			--AND DAY(@time_col_name) >= DAY(@start_time) AND
 			--DAY(@time_col_name) <= DAY(@end_time)
			--如果member_id为0，则不进行member_id的筛选
		AND
				( member_id = @member_id )
		AND 	
				(@drop_nan = 0 OR @var_col_name IS NOT NULL)
		AND 	
				(@select_positive = 0 OR @var_col_name > 0)
			)
	)
,

--定义T1为源数据+quant列
T1 AS
(
SELECT
	cesm_output1.quant quant,
	cesm_output1.percentile percentile,
	cesm_output1.event_kind event_kind,
	@percentile_period AS percentile_period,
	T0_temp.*
FROM
	(
	SELECT 
		lon,
		lat,
		time_list,
		--CONVERT(DATE, time_list, 23) 	 time_list,
		climate_index

	FROM 
		T0)T0_temp
LEFT JOIN
	(SELECT
		*
	FROM
		mean_of_cesm_output1
	WHERE 
		percentile = ROUND(@quant,2) AND
		event_kind = @event_kind AND
		member_id = @member_id
		)cesm_output1
on 
	T0_temp.lon = cesm_output1.lon AND
	T0_temp.lat = cesm_output1.lat 

),

--T2为增加一列HW,超过quant，HW=1，else 0
T2 AS (
SELECT 
	lon, 
	lat,
	time_list,
	climate_index,
	quant,
	percentile,
	percentile_period,
	event_kind,
	(CASE 
		WHEN quant <= T1.climate_index 
		THEN  1 
	ELSE 0 END) AS HW
FROM 
	T1),

--T3为取出异常值（HW=1）并且增加一列行坐标，作为后续对齐使用
T3 AS (
--取出异常温度（event）的数据
SELECT 
	lon, 
    lat,
    time_list,
    climate_index,
    quant,
    HW,
	percentile,
	percentile_period,
	event_kind,
    ROW_NUMBER() OVER ( ORDER BY lat,lon,time_list) AS row_num
FROM
	T2
WHERE 
	HW = 1
),

-----------------------------------------
T3_SHIFT AS (
    SELECT 
        lon, 
        lat,
        time_list,
        climate_index,
        quant,
        HW,
		percentile,
		percentile_period,
		event_kind,
        (
            CASE 
                WHEN DATEDIFF(time_list, LAG(time_list, 1) OVER (ORDER BY row_num),'dd') = 1 
                    AND lat = LAG(lat, 1) OVER (ORDER BY row_num) 
                    AND lon = LAG(lon, 1) OVER (ORDER BY row_num)
                THEN 0
                ELSE 1
            END
        ) AS diff,
		row_num
    FROM T3
),

 T_fina_0 AS(
 SELECT
	lon,
 	lat,
 	time_list,
 	climate_index,
 	quant,
 	HW,
	diff,
	percentile,
	percentile_period,
	event_kind,
	COUNT(group_id) OVER(PARTITION BY group_id  ORDER BY group_id) duration,
	group_id,
	row_num
 FROM 
	 (
	 SELECT 
 		lon,
 		lat,
 		time_list,
 		climate_index,
 		quant,
 		HW,
		diff,
		percentile,
		percentile_period,
		event_kind,
 		--将diff列累加，得到分组id
 		SUM(diff) OVER (ORDER BY row_num) AS group_id,
 		row_num
	 FROM
 		T3_SHIFT
	)T_temp
	
 ),

  T_final AS(
 SELECT
	lon,
 	lat,
 	time_list,
 	climate_index,
 	quant,
 	HW,
	diff,
	percentile,
	percentile_period,
	event_kind,
	COUNT(group_id) OVER(PARTITION BY group_id  ORDER BY group_id) duration,
	group_id,
	row_num
 FROM 
	T_fina_0 
where 
	duration>=@duration_threshold
	
 ),

 --计算频率
freq_col AS
(
SELECT
	lat,
	lon,
	COUNT(DISTINCT group_id)	feq --对lat,lon分组后，对出现的不重复组号计数
FROM
	T_final FT
GROUP BY
	FT.lat ,FT.lon
),

--计算持续时间
dura_col AS
(
SELECT
	 DISTINCT lat, lon, total_dur	--由于T_2表中，某个区域出现多次事件，累加后会出现重复值，因此去重
 
FROM
(
	SELECT
		lat,
		lon,
		SUM(avg_d) OVER(PARTITION BY lat,lon) total_dur 
	FROM	--表T_1为对地区的单一事件（经过group_id的分组后，将每个区域发生多次事件区分）发生的持续时间进行去重（avg）
	(
		SELECT 
			lat,
			lon,
			AVG(duration) avg_d
		FROM
			T_final FT
		GROUP BY
			FT.lat,FT.lon,group_id
	)T_1
)T_2	--T_2表对某个地区进行累加
),

--计算强度
intens_col AS(
SELECT 
	lat,
	lon,
	AVG(FT.climate_index) mean_tmp
FROM
	T_final FT
GROUP BY
	lat,
	lon
),
--指标集合
Metrics_col AS(
SELECT
	T_t.frequence,
	ic.mean_tmp mean_t,
	T_t.total_duration,
	T_t.lat,
	T_t.lon
FROM
(
	SELECT
		fc.feq frequence,
		dc.total_dur total_duration,
		fc.lat lat,
		fc.lon lon
	FROM
		freq_col fc,dura_col dc
	WHERE 
		fc.lat = dc.lat AND
		fc.lon = dc.lon
	)T_t,intens_col ic
WHERE
	T_t.lat = ic.lat AND
	T_t.lon = ic.lon
),

-----------------------------最终结果-----------------
result AS(
SELECT
	ft.lat,
	ft.lon,
	ft.time_list,
	ft.climate_index climate_index,
	ft.quant quant,
	mc.frequence,
	mc.mean_t mean_climate_index, -- mean_climate_index,
	ft.group_id,
	ft.duration single_event_dur,--某个地区单一事件的事件
	mc.total_duration total_duration,	--某个地区发生事件的总时间
	ft.percentile percentile,
	-- ft.percentile_period percentile_period,
	ft.row_num
	-- ft.event_kind event_kind

FROM
	Metrics_col mc,T_final ft
WHERE
	ft.lat = mc.lat AND ft.lon = mc.lon
)





INSERT INTO TABLE @output_table partition( percentile_period = @percentile_period,event_kind = @event_kind)
SELECT 
	CASE WHEN @member_id > 0 THEN @member_id END AS member_id,
	* except(row_num)
FROM result

ORDER BY 
		lat,
		lon,
 		time_list;
