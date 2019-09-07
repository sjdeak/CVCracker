#include "RuneDetector.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <iostream>

using namespace cv;
using namespace std;


// 二值化和轮廓检测
pair<int, int> RuneDetector::getTarget(const cv::Mat & image){
	cvtColor(image, src, CV_BGR2GRAY);
	Mat binary;
	// cv::二值化
    threshold(src, binary, 150, 255, THRESH_BINARY);

	vector<vector<Point2i>> contours;
	vector<Vec4i> hierarchy;
	
	// cv::轮廓检测
	findContours(binary, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	sudoku_rects.clear();
	if (checkSudoku(contours, sudoku_rects)){
        if (use_perspective == true){
            pair<int, int> idx = chooseTargetPerspective(src, sudoku_rects);
            return idx;
        }
        else{
            pair<int, int> idx = chooseTarget(src, sudoku_rects);
            return idx;
        }
	}
	return make_pair(-1,-1);
}

// 从轮廓检测的结果中筛选出九宫格块
bool RuneDetector::checkSudoku(const vector<vector<Point2i>> & contours, vector<RotatedRect> & sudoku_rects) {
	if (contours.size() < 9)
        return false;

    float width = sudoku_width;
	float height = sudoku_height;
	float ratio = 28.0 / 16.0;  // 标准宽高比
	int sudoku = 0;

    float low_threshold = 0.6;
    float high_threshold = 1.4;
    vector<Point2f> centers;
	for (size_t i = 0; i < consrc_csmtours.size(); i++) {
		RotatedRect rect = minAreaRect(contours[i]);
		rect = adjustRRect(rect);
		const Size2f & s = rect.size;
		float ratio_cur = s.width / s.height; // 当前轮廓的宽高比

		if (ratio_cur > 0.8 * ratio && ratio_cur < 1.2 * ratio &&  // 宽高比
			s.width > low_threshold * width && s.width < high_threshold * width &&
			s.height > low_threshold * height && s.height < high_threshold * height &&  // 实际长宽
			((rect.angle > -10 && rect.angle < 10) || rect.angle < -170 || rect.angle > 170)){  // 旋转度

			sudoku_rects.push_back(rect);
            centers.push_back(rect.center);
            ++sudoku;
		}
	}

    // 已经是非常强的筛选了

    if (sudoku > 15)
        return false;

    if(sudoku > 9){
        float dist_map[15][15] = {0};  // [i][j] -> 第i个矩形和第j个矩形间的距离
        // calculate distance of each cell center
        for(int i = 0; i < sudoku; ++i){
            for (int j = i+1; j < sudoku; ++j){
                float d = sqrt((centers[i].x - centers[j].x)*(centers[i].x - centers[j].x) + (centers[i].y - centers[j].y)*(centers[i].y - centers[j].y));
                dist_map[i][j] = d;
                dist_map[j][i] = d;
            }
        }

        // 看看哪一个矩形到其他矩形的距离之和最小，那个矩形就是九宫格中间那块
        int center_idx = 0;
        float min_dist = 100000000;
        for(int i = 0; i < sudoku; ++i){
            float cur_d = 0;
            for (int j = 0; j < sudoku; ++j){
                cur_d += dist_map[i][j];
            }
            if(cur_d < min_dist){
                min_dist = cur_d;
                center_idx = i;
            }
        }

        // sort distance between each cell and the center cell
        vector<pair<float, int> > dist_center;
        for (int i = 0; i < sudoku; ++i){
            dist_center.push_back(make_pair(dist_map[center_idx][i], i));
        }
        std::sort(dist_center.begin(), dist_center.end(), [](const pair<float, int> & p1, const pair<float, int> & p2) { return p1.first < p2.first; });

		// 贪心
        // choose the nearest 9 cell as suduku
        vector<RotatedRect> sudoku_rects_temp;
        for(int i = 0; i < 9; ++i){
            sudoku_rects_temp.push_back(sudoku_rects[dist_center[i].second]);
        }
        sudoku_rects_temp.swap(sudoku_rects);
    }
    cout << "sudoku n: " << sudoku_rects.size()  << endl;
	return sudoku_rects.size() == 9;
}

// 利用透视变换将图像变为正视图，截取出九宫格中的九个格子
pair<int, int> RuneDetector::chooseTargetPerspective(const Mat & image, const vector<RotatedRect> & sudoku_rects){
	// get 9(cell) X 4(corner) corner, and 9 cell's center
	vector<Point2fWithIdx> centers;
	vector<Point2f> corner;
	for (size_t i = 0; i < sudoku_rects.size(); i++)	{
		const RotatedRect & rect = sudoku_rects[i];
		Point2f vecs[4];
		rect.points(vecs);
		for (size_t j = 0; j < 4; j++) {
			corner.push_back(vecs[j]);  // cornor: 0(四角的点) 1(四角的点)...
		}
		centers.push_back(Point2fWithIdx(rect.center, i));
	}

	// arange sudoku cell to following order
	// 0  1  2
	// 3  4  5
	// 6  7  8
	sort(centers.begin(), centers.end(), [](const Point2fWithIdx & p1, const Point2fWithIdx & p2) { return p1.p.y < p2.p.y; });
	sort(centers.begin() + 0, centers.begin() + 3, [](const Point2fWithIdx & p1, const Point2fWithIdx & p2) { return p1.p.x < p2.p.x; });
	sort(centers.begin() + 3, centers.begin() + 6, [](const Point2fWithIdx & p1, const Point2fWithIdx & p2) { return p1.p.x < p2.p.x; });
	sort(centers.begin() + 6, centers.begin() + 9, [](const Point2fWithIdx & p1, const Point2fWithIdx & p2) { return p1.p.x < p2.p.x; });

	// get position of [0,2,6,8] corner  四个角
	int corner_idx[] = { 0, 2, 6, 8 };
	vector<Point2f> corner_0268;
	for (size_t i: {0, 2, 6, 8}) {
		size_t k = centers[i].idx * 4;  // centers: vector<Point2fWithIdx>
		for (size_t j = 0; j < 4; j++){
			corner_0268.push_back(corner[k + j]);  // corner_0268: 0(四角的点), 2(四角的点)...
		}
	}

	// find approx corner of sudoku
	RotatedRect rect = minAreaRect(corner_0268);
	Point2f vertices[4];
	rect.points(vertices);
	Point2f lu, ld, ru, rd;  // 四个角点
	sort(vertices, vertices + 4, [](const Point2f & p1, const Point2f & p2) { return p1.x < p2.x; });
	if (vertices[0].y < vertices[1].y){
		lu = vertices[0];
		ld = vertices[1];
	}
	else{
		lu = vertices[1];
		ld = vertices[0];
	}
	if (vertices[2].y < vertices[3].y)	{
		ru = vertices[2];
		rd = vertices[3];
	}
	else {
		ru = vertices[3];
		rd = vertices[2];
	}

	// find actual corner of sudoku
	Point2f _lu, _ld, _ru, _rd;
	float mlu = 10000.0, mld = 10000.0, mru = 10000.0, mrd = 10000.0;
	for (size_t i = 0; i < corner_0268.size(); i++) {
		const Point2f & p = corner_0268[i];
		float v1 = (p - lu).dot((p - lu));
		float v2 = (p - ld).dot((p - ld));
		float v3 = (p - ru).dot((p - ru));
		float v4 = (p - rd).dot((p - rd));
		if (v1 < mlu) {
			mlu = v1;
			_lu = p;
		}
		if (v2 < mld) {
			mld = v2;
			_ld = p;
		}
		if (v3 < mru) {
			mru = v3;
			_ru = p;
		}
		if (v4 < mrd) {
			mrd = v4;
			_rd = p;
		}
	}

	// applies a perspective transformation to an image
	
	// 计算整个九宫格的长宽
	float _width = max((_lu - _ru).dot(_lu - _ru), (_ld - _rd).dot(_ld - _rd));
	float _height = max((_lu - _ld).dot(_lu - _ld), (_rd - _ru).dot(_rd - _ru));
	_width = sqrtf(_width);
	_height = sqrtf(_height);

	vector<Point2f> src_p;
	src_p.push_back(_lu);
	src_p.push_back(_ld);
	src_p.push_back(_ru);
	src_p.push_back(_rd);

	vector<Point2f> dst_p;
	dst_p.push_back(Point2f(0.0, 0.0));
	dst_p.push_back(Point2f(0.0, _height));
	dst_p.push_back(Point2f(_width, 0.0));
	dst_p.push_back(Point2f(_width, _height));

	Mat perspective_mat = getPerspectiveTransform(src_p, dst_p);  // 透视矩阵H
	Mat image_persp;
	warpPerspective(image, image_persp, perspective_mat, Size(_width, _height));

	// 计算单个格子的长宽
	const double * pdata = (double *)perspective_mat.data;
	float height_avg = 0.0, width_avg = 0.0;
	
	for (size_t i = 0; i < sudoku_rects.size(); ++i) {
		vector<Point2f> vec_p; // vec_p 单个格子的四个角
		for (size_t j = 0; j < 4; j++) {
			const Point2f & p = corner[i * 4 + j];
			float x = pdata[0] * p.x + pdata[1] * p.y + pdata[2];
			float y = pdata[3] * p.x + pdata[4] * p.y + pdata[5];
			float s = pdata[6] * p.x + pdata[7] * p.y + pdata[8];
			vec_p.push_back(Point2f(x / s, y / s));
		}
		Rect2f r = boundingRect(vec_p);
		height_avg += r.height;
		width_avg += r.width;
	}
	height_avg /= 9.0;
	width_avg /= 9.0;

    if(height_avg > _height / 3)
        height_avg = 0.25 * _height;
    if(width_avg > _width / 3)
        width_avg = 0.25 * _width;

	// 到这里我们就得到了九个九宫格的确切图像, 接下来计算ORB特征, 找出目标图像
    int cell_width = 0.48 * width_avg + 0.5;
    int cell_height = 0.50 * height_avg + 0.5;
	int half_w_gap = (width_avg - cell_width) / 2, half_h_gap = (height_avg - cell_height) / 2;  // gap: 格子间的黑边
    int offset_x = 0.05 * cell_width + 0.5;  // 略微割内部一些，以保证没有边缘黑色残留
    int offset_y = 0.05 * cell_height + 0.5;
	int width_start[] = { half_w_gap, (_width - cell_width) / 2, _width - cell_width - half_w_gap };
    int height_start[] = { half_h_gap, (_height - cell_height) / 2, _height - cell_height - half_h_gap };

	Mat cell[9];
	for (size_t i = 0; i < 3; i++){
		for (size_t j = 0; j < 3; j++){
			size_t idx = i * 3 + j;
            Rect cell_roi(width_start[j]+offset_x, height_start[i]+offset_y, cell_width, cell_height);  // roi: region of interst
			image_persp(cell_roi).copyTo(cell[idx]);
		}
	}

    int idx = -1;
    if (type == RUNE_ORB)
        idx = findTargetORB(cell);
    else if (type == RUNE_GRAD)
        idx = findTargetEdge(cell);
    else if (type = RUNE_CANNY)
        idx = findTargetCanny(cell);
}

// 对九个格子进行特征提取和匹配，找出目标在九宫格中哪个位置
// 利用ORB特征，进行特征匹配，匹配度最低的cell为目标
int RuneDetector::findTargetORB(cv::Mat * cells){  // cells 九个格子
	Mat descriptor[9];
	vector<vector<KeyPoint> > keypoints;
	keypoints.resize(9);
	Ptr<ORB> orb = cv::ORB::create(100, 1, 1, 10, 0, 2, 1, 17);
	BFMatcher matcher(NORM_HAMMING, 0);  // cv::暴力匹配器
	int match_count[9][9] = { { 0 } };

	for (size_t i = 0; i < 9; i++)	{
		vector<KeyPoint> & kp = keypoints[i];
		Mat & desp = descriptor[i];
		orb->detectAndCompute(cells[i], Mat(), kp, desp);

        if (desp.rows < 2)
            return -1;

		// feature matching
		for (size_t k = 0; k < i; k++){
			vector<vector<DMatch> > matches;
			matcher.knnMatch(desp, descriptor[k], matches, 2);
			int cnt = 0;
			for (size_t n = 0; n < matches.size(); n++)	{
				vector<DMatch> & m = matches[n];
				DMatch & dm1 = m[0];
				DMatch & dm2 = m[1];
				if (dm1.distance / dm2.distance < 0.8){
					cnt++;
				}
			}
			match_count[i][k] = cnt;
			match_count[k][i] = cnt;
		}
	}

    // choose the minimun match cell as the target
	float avg_cnt[9] = {};
	int min_idx = -1;
	float min_cnt = 65535;
	for (size_t i = 0; i < 9; i++){
		for (size_t j = 0; j < 9; j++){
			avg_cnt[i] += match_count[i][j];
		}
		if (avg_cnt[i] < min_cnt){
			min_cnt = avg_cnt[i];
			min_idx = i;
		}
	}
	return min_idx;
}