import re
def fix_facilities(facilities):
    # Sử dụng regex để tìm các từ khóa bắt đầu bằng chữ hoa
    pattern = r'\b[A-Z]\w*'
    facilities_list = re.findall(pattern, facilities)
    if "Hồ" in facilities or "hồ" in facilities:
        facilities_list[0] = "Hồ bơi ngoài trời"
    else:
        facilities_list[0] = "2 hồ bơi"
    # Xử lý các trường hợp đặc biệt
    for i, facility in enumerate(facilities_list):
        if facility == 'WiFi':
            facilities_list[i] = 'WiFi miễn phí'
        elif facility == 'Sân':
            facilities_list[i] = 'Sân thượng / hiên'
        elif facility == 'Bữa':
            if 'tuyệt' in facilities:
                facilities_list[i] = 'Bữa sáng tuyệt hảo'
            else:
                facilities_list[i] = 'Bữa sáng rất tốt'
        elif facility == 'Xe':
            facilities_list[i] = 'Xe đưa đón sân bay'
        elif facility == 'Pho':
            if 'Phòng gia đình' in facilities_list:
                facilities_list[i] = 'Phòng không hút thuốc'
            else:
                facilities_list[i] = 'Phòng gia đình'
        elif facility == 'Trung':
            if facilities_list[i+1] == 'Spa' and i + 1 < len(facilities_list):
                    facilities_list[i] = 'Trung tâm Spa & chăm sóc sức khoẻ'
                    if i + 1 < len(facilities_list):
                        facilities_list.pop(i + 1)
            else:
                facilities_list[i] = 'Trung tâm thể dục'
        elif facility == 'Chỗ' or facility == 'Chô':
            if '(trong khuôn viên)' in facilities:
                facilities_list[i] = 'Chỗ đậu xe (trong khuôn viên)'
            else:
                facilities_list[i] = 'Chỗ đỗ xe miễn phí'
        elif facility == 'Quầy':
            facilities_list[i] = 'Quầy bar'
        elif facility == 'Giáp':
            facilities_list[i] = 'Giáp biển'
        elif facility == 'Khu':
            facilities_list[i] = 'Khu vực bãi tắm riêng'
        elif facility == 'Lê' or facility == 'Lễ':
            facilities_list[i] = 'Lễ tân 24h' 
        elif facility == 'Nha' or facility == 'Nhà':
            facilities_list[i] = 'Nhà hàng'
        elif facility == 'Dịch':
            facilities_list[i] = 'Dịch vụ đưa đón sân bay (miễn phí)'
        elif facility == 'Di':
            facilities_list[i] = 'Dịch vụ phòng'
        elif facility == 'Điều':
            facilities_list[i] = 'Điều hòa nhiệt độ' 
        elif facility == 'Máy':
            facilities_list[i] = 'Máy pha trà/cà phê trong tất cả các phòng'
    return facilities_list

def fix_subscore(review_subscore):
    result = {}
    for item in review_subscore:
        match = re.match(r'(.*?)(\d+,?\d+)', item)
        if match:
            key, value = match.groups()
            value = value.replace(',', '.')
            result[key.strip()] = float(value)
    return result

def fix_address(address):
    result = re.split(r',', address)[0]
    return result

def fix_scores(scores):
    result = scores.split(" ")[-1]
    return result

def fix_link(u, to="vi"):
    id = u.find(".html")
    if to == "vi":
        if u[int(id - 5):int(id)] != "en-gb" and u[int(id - 2):int(id)] != "vi":
            return u.replace(".html", ".vi.html")
        else:
            return u.replace("en-gb.html", "vi.html")
    elif to == "en":
        if u[int(id - 2):int(id)] == "vi":
            return u.replace("vi.html", "en-gb.html")