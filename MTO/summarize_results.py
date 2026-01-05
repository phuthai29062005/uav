import os

def summarize_gen_999(root_folder="results", output_file="summary_gen_999.txt"):
    # Danh sách để chứa kết quả tạm
    data_list = []
    
    print("Đang quét dữ liệu...")

    # Duyệt qua cây thư mục
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            # Chỉ xử lý các file task*.txt (ví dụ task1.txt, task2.txt...)
            if file.startswith("task") and file.endswith(".txt"):
                file_path = os.path.join(subdir, file)
                
                # Phân tích đường dẫn để lấy Seed và Check
                # Ví dụ path: results/seed0/check_50/task1.txt
                parts = file_path.split(os.sep)
                
                seed_name = "unknown"
                check_name = "unknown"
                
                # Tìm tên seed và check trong đường dẫn
                for part in parts:
                    if "seed" in part:
                        seed_name = part
                    if "check_" in part:
                        check_name = part
                
                task_name = file.replace(".txt", "") # Lấy tên task (bỏ đuôi .txt)
                
                # Đọc file để tìm Gen 999
                fitness_999 = "N/A" # Giá trị mặc định nếu không tìm thấy
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # Bỏ qua dòng đầu tiên (header)
                        for line in lines[1:]:
                            content = line.strip().split()
                            if len(content) >= 2:
                                gen = content[0]
                                fitness = content[1]
                                
                                # Kiểm tra xem có phải gen 999 không (so sánh chuỗi hoặc số)
                                if gen == "999" or gen == "999.0":
                                    fitness_999 = fitness
                                    break
                except Exception as e:
                    print(f"Lỗi đọc file {file_path}: {e}")

                # Lưu vào danh sách
                data_list.append({
                    "seed": seed_name,
                    "check": check_name,
                    "task": task_name,
                    "fitness": fitness_999
                })

    # Sắp xếp lại danh sách cho dễ nhìn (theo Seed -> Check (số) -> Task (số))
    def sort_key(item):
        # Trích xuất số từ check_50 để sort đúng (50 < 100) thay vì sort chuỗi
        try:
            check_num = int(item["check"].split('_')[1])
        except:
            check_num = 0
            
        try:
            task_num = int(item["task"].replace('task', ''))
        except:
            task_num = 0
            
        return (item["seed"], check_num, task_num)

    data_list.sort(key=sort_key)

    # Ghi ra file tổng hợp
    with open(output_file, 'w', encoding='utf-8') as out:
        # Ghi tiêu đề cột
        header = f"{'Seed':<10} | {'Check':<12} | {'Task':<10} | {'Fitness Gen 999':<20}\n"
        out.write(header)
        out.write("-" * len(header) + "\n")
        
        for item in data_list:
            line = f"{item['seed']:<10} | {item['check']:<12} | {item['task']:<10} | {item['fitness']:<20}\n"
            out.write(line)

    print(f"Xong! Đã tạo file '{output_file}' với toàn bộ dữ liệu.")

if __name__ == "__main__":
    summarize_gen_999()