import math
import numpy as np
from scipy.spatial.transform import Rotation as Rot

"""
aiming for transit the extrinsic euler to quaternion out of 
#!!! 2024-Jun 's version calibration result for rotation part:"""
F120 = {
'fxn': 0.7478020191192627,
'fyn': 0.39888936281204224,
'fzn': 1.69412100315094,
'fxc': 0.12453147768974304,
'fyc': 0.023205328732728958,
'fzc': 0.17563281953334808,
}     


FL120 = {
'fxn': 0.6823626160621643,
'fyn': 0.6535221934318542,
'fzn': 67.79759216308594,
'fxc': -0.0029870772268623114,
'fyc': -0.574974536895752,
'fzc': -0.31926506757736206,
}   

FR120 = {
'fxn': 0.5174974799156189,
'fyn': -0.8920024633407593,
'fzn': -67.4144287109375,
'fxc': 0.7106829285621643,
'fyc': 0.8196682929992676,
'fzc': 0.5588372945785522,
} 

RL = {
'fxn': 0.8304807543754578,
'fyn': -0.27166205644607544,
'fzn': 151.30245971679688,
'fxc': -0.39371949434280396,
'fyc': 0.16318558156490326,
'fzc': 0.4732438027858734,
}

RR = {
'fxn': 1.896492600440979,
'fyn': -0.2010795921087265,
'fzn': -151.93609619140625,
'fxc': -0.18918843567371368,
'fyc': 0.7620489597320557,
'fzc': 0.1966867595911026
}


R30 = {
'fxn': -2.2612860202789307,
'fyn': -1.897634506225586,
'fzn': 178.9329376220703,
'fxc': -0.28133028745651245,
'fyc': 0.5165433287620544,
'fzc': 0.10230113565921783
}

# !!! 2024-Feb 's version calibration result for rotation part:
# F120 = {
# 'fxn': 0.9732222557067871,
# 'fyn': 0.13335508108139038,
# 'fzn': 2.095618486404419,
# 'fxc': -0.07370994240045547,
# 'fyc': 0.024520529434084892,
# 'fzc': -0.17564187943935394,
# }     

# FL120 = {
# 'fxn': 0.7091734409332275,
# 'fyn': 0.3381761312484741,
# 'fzn': 67.89298248291016,
# 'fxc': -0.12577678263187408,
# 'fyc': 0.23768751323223114,
# 'fzc': -0.5289285778999329,
# }   

# FR120 = {
# 'fxn': 1.0962799787521362,
# 'fyn': -0.33374258875846863,
# 'fzn': -66.79678344726562,
# 'fxc': -0.03324808180332184,
# 'fyc': -0.014599476009607315,
# 'fzc': -0.07454647868871689,
# } 

# RL = {
# 'fxn': 0.4788847267627716,
# 'fyn': 0.16447968780994415,
# 'fzn': 151.3782958984375,
# 'fxc': 0.24541577696800232,
# 'fyc': 0.33335253596305847,
# 'fzc': 0.024203738197684288,
# }


# RR = {
# 'fxn': 1.3843307495117188,
# 'fyn': 0.4358459413051605,
# 'fzn': -151.0899658203125,
# 'fxc': 0.0210320632904768,
# 'fyc': 0.17808659374713898,
# 'fzc': -0.47936850786209106
# }

# R30 = {
# 'fxn': -2.7012135982513428,
# 'fyn': -0.6712470650672913,
# 'fzn': 179.31748962402344,
# 'fxc': 0.2679538130760193,
# 'fyc': 0.2423640340566635,
# 'fzc': -0.2659626007080078
# }
    
def quaternion_to_euler(q):  
    # Normalize the quaternion  
    norm = math.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)  
    q = [i/norm for i in q]  
      
    # Calculate roll (x-axis rotation)  
    sinr_cosp = 2.0 * (q[0]*q[1] + q[2]*q[3])  
    cosr_cosp = 1.0 - 2.0 * (q[1]**2 + q[2]**2)  
    roll = math.atan2(sinr_cosp, cosr_cosp)  
      
    # Calculate pitch (y-axis rotation)  
    sinp = 2.0 * (q[0]*q[2] - q[3]*q[1])  
    if abs(sinp) >= 1:  
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range  
    else:  
        pitch = math.asin(sinp)  
      
    # Calculate yaw (z-axis rotation)  
    siny_cosp = 2.0 * (q[0]*q[3] + q[1]*q[2])  
    cosy_cosp = 1.0 - 2.0 * (q[2]**2 + q[3]**2)  
    yaw = math.atan2(siny_cosp, cosy_cosp)  
    roll = math.degrees(roll)  
    pitch = math.degrees(pitch)  
    yaw = math.degrees(yaw)  
      
    return [roll,pitch,yaw]  
  
    
def euler_to_quaternion(roll, pitch, yaw):

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp
    return w, x, y, z


    #Convert Euler angles to rotation matrix
def iso8855_to_nuscenes_claude(roll, pitch, yaw):
    R_camera = np.array([[np.cos(yaw) * np.cos(pitch),
    np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
    np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
    [np.sin(yaw) * np.cos(pitch),
    np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
    np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
    [-np.sin(pitch),
    np.cos(pitch) * np.sin(roll),
    np.cos(pitch) * np.cos(roll)]])
    #Transformation matrix to convert from your camera coordinate system to NuScenes coordinate system
    T_camera_to_nuscenes = np.array([[0, -1, 0],
    [0, 0, -1],
    [1, 0, 0]])
    #Apply the transformation
    R_nuscenes = T_camera_to_nuscenes @ R_camera
    #Convert the transformed rotation matrix to Euler angles
    yaw_nuscenes = np.arctan2(R_nuscenes[1, 0], R_nuscenes[0, 0])
    pitch_nuscenes = np.arcsin(-R_nuscenes[2, 0])
    roll_nuscenes = np.arctan2(R_nuscenes[2, 1], R_nuscenes[2, 2])
    #Convert the Euler angles to quaternion
    w = np.cos(yaw_nuscenes/2) * np.cos(pitch_nuscenes/2) * np.cos(roll_nuscenes/2) + np.sin(yaw_nuscenes/2) * np.sin(pitch_nuscenes/2) * np.sin(roll_nuscenes/2)
    x = np.sin(yaw_nuscenes/2) * np.cos(pitch_nuscenes/2) * np.cos(roll_nuscenes/2) - np.cos(yaw_nuscenes/2) * np.sin(pitch_nuscenes/2) * np.sin(roll_nuscenes/2)
    y = np.cos(yaw_nuscenes/2) * np.sin(pitch_nuscenes/2) * np.cos(roll_nuscenes/2) + np.sin(yaw_nuscenes/2) * np.cos(pitch_nuscenes/2) * np.sin(roll_nuscenes/2)
    z = np.cos(yaw_nuscenes/2) * np.cos(pitch_nuscenes/2) * np.sin(roll_nuscenes/2) - np.sin(yaw_nuscenes/2) * np.sin(pitch_nuscenes/2) * np.cos(roll_nuscenes/2)
    print(f"Quaternion representation: [w, x, y, z] = [{w}, {x}, {y}, {z}]")
    quaternion = np.array([w, x, y, z])
    return quaternion

def iso8855_to_nuscenes_4o(roll, pitch, yaw):
    # Define the extrinsic parameters in radians
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    
    # Create a rotation object from roll, pitch, yaw (ISO 8855 to your camera's coordinate system)
    r_iso = Rot.from_euler('xyz', [roll, pitch, yaw])
    
    # Define the transformation from ISO 8855 to NuScenes:
    # Rotate yaw by -90 degrees and roll by -90 degrees
    R_z = Rot.from_euler('z', -90, degrees=True)
    R_x = Rot.from_euler('x', -90, degrees=True)
    
    # Combine the transformations
    combined_rotation = R_z * R_x
    
    # Apply the combined rotation to the original extrinsics
    transformed_rotation = combined_rotation * r_iso
    
    # Convert the resulting rotation to a quaternion
    quaternion = transformed_rotation.as_quat()  # [x, y, z, w]
    
    # Adjust the quaternion format to [w, x, y, z]
    quaternion = np.roll(quaternion, shift=1)
    
    return quaternion

def iso8855_to_nuscenes_lama(extrinsic_params):
    # Define the rotation matrices for the ISO 8855 coordinate system
    Rx = np.array([[1, 0, 0],
    [0, np.cos(extrinsic_params['roll']), -np.sin(extrinsic_params['roll'])],
    [0, np.sin(extrinsic_params['roll']), np.cos(extrinsic_params['roll'])]])
    Ry = np.array([[np.cos(extrinsic_params['pitch']), 0, np.sin(extrinsic_params['pitch'])],
                [0, 1, 0],
                [-np.sin(extrinsic_params['pitch']), 0, np.cos(extrinsic_params['pitch'])]])

    Rz = np.array([[np.cos(extrinsic_params['yaw']), -np.sin(extrinsic_params['yaw']), 0],
                [np.sin(extrinsic_params['yaw']), np.cos(extrinsic_params['yaw']), 0],
                [0, 0, 1]])

    # Convert to NuScenes' coordinate system
    T = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    R_nuscenes = T @ Rz @ Ry @ Rx

    # Convert to quaternion representation
    w = 0.5 * np.sqrt(1 + R_nuscenes[0, 0] + R_nuscenes[1, 1] + R_nuscenes[2, 2])
    x = (R_nuscenes[2, 1] - R_nuscenes[1, 2]) / (4 * w)
    y = (R_nuscenes[0, 2] - R_nuscenes[2, 0]) / (4 * w)
    z = (R_nuscenes[1, 0] - R_nuscenes[0, 1]) / (4 * w)

    quaternion = np.array([w, x, y, z])
    return quaternion

def calculate_values(data_structs):  
    result = {}  
    result_for_print = {}
    i=0
    for data_struct in data_structs:  
        fxn = data_struct['fxn']  
        fyn = data_struct['fyn']  
        fzn = data_struct['fzn']  
        fxc = data_struct['fxc']  
        fyc = data_struct['fyc']  
        fzc = data_struct['fzc']  

        # result[i] = {  
        #     'fxn_plus_fxc': fxn + fxc,  
        #     'fyn_plus_fyc': fyn + fyc,  
        #     'fzn_plus_fzc': fzn + fzc  
        # }
        
        result_for_print[i] = {  
            'fxn_plus_fxc': (fxn + fxc) - 90,  
            'fyn_plus_fyc': (fyn + fyc),  
            'fzn_plus_fzc': (fzn + fzc) - 90
        }
        print(f"===================euler of {i}'s view below: =================")
        print(result_for_print[i])
        """
        ===================euler of 0's view below: =================
        {'fxn_plus_fxc': 0.5833966583013535, 'fyn_plus_fyc': 0.5758636444807053, 'fzn_plus_fzc': 67.36405390501022}
        ===================euler of 1's view below: =================
        {'fxn_plus_fxc': 0.8995123133063316, 'fyn_plus_fyc': 0.15787561051547527, 'fzn_plus_fzc': 1.919976606965065}
        ===================euler of 2's view below: =================
        {'fxn_plus_fxc': 1.0630318969488144, 'fyn_plus_fyc': -0.34834206476807594, 'fzn_plus_fzc': -66.87132992595434}
        ===================euler of 3's view below: =================
        {'fxn_plus_fxc': 0.7243005037307739, 'fyn_plus_fyc': 0.4978322237730026, 'fzn_plus_fzc': 151.40249963663518}
        ===================euler of 4's view below: =================
        {'fxn_plus_fxc': -2.4332597851753235, 'fyn_plus_fyc': -0.42888303101062775, 'fzn_plus_fzc': 179.05152702331543}
        ===================euler of 5's view below: =================
        {'fxn_plus_fxc': 1.4053628128021955, 'fyn_plus_fyc': 0.6139325350522995, 'fzn_plus_fzc': -151.5693343281746}
        """
        result[i] = {  
            'fxn_plus_fxc': math.radians(fxn + fxc) - math.pi/2,  
            'fyn_plus_fyc': math.radians(fyn + fyc),  
            'fzn_plus_fzc': math.radians(fzn + fzc) - math.pi/2,
        }
        i+=1  
    return result
# print(euler_to_quaternion(math.pi/3,0,math.pi/6))
# #(0.8365163037378079, 0.4829629131445341, 0.12940952255126034, 0.2241438680420134)
 
# print(tfs.euler.euler2quat(math.pi/3,0,math.pi/6,"sxyz"))
# #[0.8365163  0.48296291 0.12940952 0.22414387]
 
# print(euler_to_quaternion(math.pi/3,math.pi,math.pi/2))
# #(0.3535533905932738, -0.6123724356957946, 0.6123724356957946, -0.3535533905932737)
 
# print(tfs.euler.euler2quat(math.pi/3,math.pi,math.pi/2,"sxyz"))
# #array([ 0.35355339, -0.61237244,  0.61237244, -0.35355339])

#

if __name__ == "__main__":
    # test quaternion
    
    
    
    print("===================test-below=================")
    print("===================iso8855_to_nuscenes_lama=================")
    # Feb-2024 quaternion [0.6975, -0.6891, 0.14156, -0.136008]
    # Feb-latest (w0.6978901794028058, x-0.6888857279913172, y0.1412359076855683, z-0.1357872219438409)
    
    # raw extrinisc param of FL120
    #extrinsic_params_4o = (0.6823626160621643,
                        # 0.6535221934318542,
                        # 67.79759216308594)
    # raw extrinisc param of F120
    # extrinsic_params_4o = (0.7478020191192627, 0.39888936281204224, 1.69412100315094)
    # 
    # extrinsic_params_lama = {"roll": 0.7478020191192627, "pitch": 0.39888936281204224, "yaw": 1.69412100315094}
    # quaternion_lama = iso8855_to_nuscenes_lama(extrinsic_params_lama)
    # print(f"quaternion_lama: {quaternion_lama}")
    # print(f"quaternion_to_euler_lama: {quaternion_to_euler(quaternion_lama)}")
    # quaternion_to_euler(quaternion_lama)
    # quaternion_lama: [ w0.39514975  x0.19965813  y0.87610681 z-0.19086686]
    #quaternion_4o = iso8855_to_nuscenes_4o(extrinsic_params_4o[0], extrinsic_params_4o[1], extrinsic_params_4o[2])
    #print(f"iso8855_to_nuscenes_4o: {quaternion_4o}" )
    #print(f"quaternion_to_euler_4o: {quaternion_to_euler(quaternion_4o)}")

    # iso8855_to_nuscenes_4o: [ w0.69070121 x-0.1312589  y 0.69699404 z-0.1410753 ]
    
    # print(f"iso8855_to_nuscenes_claude_quater: {iso8855_to_nuscenes_claude(extrinsic_params_4o[0], extrinsic_params_4o[1], extrinsic_params_4o[2])}" )
    # print(f"quaternion2euler_claude: {quaternion_to_euler(iso8855_to_nuscenes_claude(extrinsic_params_4o[0], extrinsic_params_4o[1], extrinsic_params_4o[2]))}")

    

    # # Nuscene ego to front cam, 1.yaw -90, 2.roll -90, pitch0,  
    # (0.5003981633553667, -0.49999984146591736, 0.49960183664463337, -0.49999984146591736)
    # print(euler_to_quaternion(-3.14/2, 0, -3.14/2))
    
    
    # (0.5003981633553667, -0.49999984146591736, -0.49960183664463337, 0.49999984146591736)
    # print(euler_to_quaternion(-3.14/2, 0, 3.14/2))
    
    # # Nuscene ego to rear cam, 1.roll -90, 2.pitch +90, 
    # # self(0.5003981633553667, -0.49999984146591736, 0.49999984146591736, 0.49960183664463337)
    # # nuscene q: [0.5037872666382278,-0.49740249788611096,-0.4941850223835201,0.5045496097725578],
    # print(euler_to_quaternion(-3.14/2, 3.14/2, 0))
    #Nuscene ego to rear cam, 1.yaw 90 , 2. roll -90 
    #nuscene q: [0.5037872666382278,-0.49740249788611096,-0.4941850223835201,0.5045496097725578],
    
    # print(euler_to_quaternion(0.01-3.14/2, -0.01, 3.1-3.14/2)) # MB2N matched
    # (0.5003981633553667, -0.49999984146591736, -0.49960183664463337, 0.49999984146591736)
    
    # #Nuscene ego to rear cam, 1.yaw 90 , 2. roll -90 
    # #nuscene q: [0.5037872666382278,-0.49740249788611096,-0.4941850223835201,0.5045496097725578],
    
    # print(euler_to_quaternion(-3.14/2, 0, -3.14/2))
    # (0.5003981633553667, -0.49999984146591736, 0.49960183664463337, -0.49999984146591736)
    # #Nusc : "rotation": [0.4998015430569128,-0.5030316162024876,0.4997798114386805,-0.49737083824542755],
    
    # #test rig2camback
    
    # print(euler_to_quaternion(0, 0, 3.14))
    # (0.0007963267107332633, 0.0, 0.0, 0.9999996829318346) #matched
    
    # #test rig2camfront
    
    # print(euler_to_quaternion(0.01-3.14/2 , 0.01, 0.01-3.14/2))
    # (1, 0.0, 0.0, 0) #matched
    #Nusc : "rotation": [0.4998015430569128,-0.5030316162024876,0.4997798114386805,-0.49737083824542755],
    
    # print(euler_to_quaternion(math.pi/3,0,math.pi/6))
    # radian_roll = math.radians(1.0962799787521362)
    # radian_pitch = math.radians(-0.033374258875846863)
    # radian_yaw = math.radians(-66.79678344726562)
    # q = Nusc_front_left
    # q = [0.6757265034669446,
    # -0.6736266522251881,
    # 0.21214015046209478,
    # -0.21122827103904068]  
    # euler_angles = quaternion_to_euler(q)  
    # print(euler_angles)
    # # Nusc_front_left : -89.85977500320001, 0.12143609391200118, -34.839035595600016
        
    # # q = Nusc_front
    # q = [0.4998015430569128,
    # -0.5030316162024876,
    # 0.4997798114386805,
    # -0.49737083824542755]  
    # euler_angles = quaternion_to_euler(q)  
    # print(euler_angles)  
    # # Nusc_front : -90.32322642770004, -0.04612719483860205, -89.6742843141]
        
    # # q = Nusc_front_right
    # q = [0.2060347966337182,
    # -0.2026940577919598,
    # 0.6824507824531167,
    # -0.6713610884174485]  
    # euler_angles = quaternion_to_euler(q)  
    # print(euler_angles)
    # # Nusc_front_right : -90.78202358850001, 0.5188438566960037, -146.404397903
        
    # # q = Nusc_back_left
    # q = [0.6924185592174665,
    # -0.7031619420114925,
    # -0.11648342771943819,
    # 0.11203317912370753]  
    # euler_angles = quaternion_to_euler(q)  
    # print(euler_angles)
    # # Nusc_back_left : -90.91736319750001, -0.21518275753700122, 18.600246142799996]
        
    # # q = Nusc_back_right
    # q = [0.12280980120078765,
    # -0.132400842670559,
    # -0.7004305821388234,
    # 0.690496031265798]  
    # euler_angles = quaternion_to_euler(q)  
    # print(euler_angles)
    # # Nusc_back_right : -90.93206677999999, 0.6190947610589997, 159.200715506]   
        
    # # q = Nusc_back
    # q = [0.5037872666382278,
    # -0.49740249788611096,
    # -0.4941850223835201,
    # 0.5045496097725578]  
    # euler_angles = quaternion_to_euler(q)  
    # print(euler_angles)
    print("===================test-above=================")
    #input
    data_structs = [FL120, F120, FR120, RL, R30, RR]  
    result = calculate_values(data_structs)  
    print(f"===================radians of each view as below: =================")
    print(result)
    
    print(f"===================quarternion of each view as below: =================")
    quaternion_halfpi = []
    for re in result:
        """converted quarternion"""
        qi = euler_to_quaternion(result[re]['fxn_plus_fxc'], result[re]['fyn_plus_fyc'], result[re]['fzn_plus_fzc'])
        print(qi)
        quaternion_halfpi.append(qi)
    
    # q = Nusc_front_left
    print("===================Verify with Nusc_front_left=================")

    # q = [0.6975617366053051, -0.6891072365851537, 0.14156435048306887, -0.1360087305376774]  
    euler_angles = quaternion_to_euler(quaternion_halfpi[0])  
    print(euler_angles)
    # Nusc_front_left : -89.85977500320001, 0.12143609391200118, -34.839035595600016
    print("Nusc_front_left : -89.85977500320001, 0.12143609391200118, -34.839035595600016 " )
    # MB -89.41660334169865, 0.5758636444807045, -22.635946094989784 #match   matched but the orien angle(67deg) diff about 12 degree
    
    # q = Nusc_front
    print("===================Verify with Nusc_front=================")

    # q = [0.5129526753793853, -0.5036182188418824, 0.4883843938307295, -0.49470084529364977]  
    euler_angles = quaternion_to_euler(quaternion_halfpi[1])  
    print(euler_angles)  
    print("Nusc_front : -90.32322642770004, -0.04612719483860205, -89.6742843141]")
    # MB -89.10048768669368, 0.15787561051547427, -88.08002339303495 #match
    
    print("===================Verify with Nusc_front_right=================")

    # q = Nusc_front_right
    q = [0.14097434541287887, -0.14255595767455692, 0.6858584396576288, -0.699572893712516]
    # result = [0.14097434541287887, -0.14255595767455692, 0.6858584396576288, -0.699572893712516]  
    euler_angles = quaternion_to_euler(q)  
    print(euler_angles)
    print("Nusc_front_right : -90.78202358850001, 0.5188438566960037, -146.404397903")
    # -88.93696810305119, -0.34834206476807944, -156.8713299259543  #match   matched but the orien angle(67deg) diff about 12 degree
    
    print("===================Verify with Nusc_back_left=================")
    # q = Nusc_back_left
    # q = [0.6102660689569468, -0.6057169867235729, -0.3560710949285466, 0.3659175213497381]  
    euler_angles = quaternion_to_euler(quaternion_halfpi[3])  
    print(euler_angles)
    print("Nusc_back_left : -90.91736319750001, -0.21518275753700122, 18.600246142799996]")
    # MB -89.27569949626923, 0.49783222377300884, 61.4024996366352 # matched but the orien angle(60deg) diff about 40 degree

    print("===================Verify with Nusc_back=================")
    # q = Nusc_back 
    # q = [0.4951951927279397, -0.5128918983128626, -0.5081034550556173, 0.48327476729569796]  
    q = [0.4989516908239229, -0.5093755950375766, -0.5125536953264245, 0.47840612748091993]
    euler_angles = quaternion_to_euler(q)  
    print(euler_angles)
    print("Nusc_back : -89.0405962694, 0.22919685786400154, 89.86124500000001] ")
    #MB -92.43325978517534, -0.42888303101061875, 89.05152702331544 #match 
    
    print("===================Verify with Nusc_back_right=================")
    
    # q = Nusc_back_right
    # q = [-0.3630978950590552, 0.3607299713498126, 0.5980064525714099, -0.6167836647128428]  
    euler_angles = quaternion_to_euler(quaternion_halfpi[5])  
    print(euler_angles)
    print("Nusc_back_right : -90.93206677999999, 0.6190947610589997, 159.200715506]   #20deg")
    #MB -88.59463718719782, 0.6139325350522989, 118.43066567182545 #match # matched but the orien angle(60deg) diff about 40 degree
    
    print("===================Verify with Nusc finished as above=================")
    
    """===================Quaternion OUTPUT below ! :================="""
    
    #Jun2024 keep the transformation(pi/2, 0, pi/2)
    (0.6976870255754957, -0.6892747898618695, 0.1377379343206684, -0.13842449152360486)
    (0.5137386524326415, -0.5023802729306853, 0.4899000063137047, -0.4936442465790682)
    (0.14292705155138524, -0.14076100075173134, 0.6851782790054733, -0.7002063442880939)
    (0.6094709956757004, -0.6041580114180398, -0.36219011577793603, 0.3638083598575427)
    (0.4989516908239229, -0.5093755950375766, -0.5125536953264245, 0.47840612748091993)
    (-0.3601890836015737, 0.3655220036511361, 0.5928761255253368, -0.6206088854368607)
    # Feb scrambled MB cood sys ext param
    (0.6975617366053051, -0.6891072365851537, 0.14156435048306887, -0.1360087305376774)
    (0.5129526753793853, -0.5036182188418824, 0.4883843938307295, -0.49470084529364977)
    (0.14097434541287887, -0.14255595767455692, 0.6858584396576288, -0.699572893712516)
    (0.6102660689569468, -0.6057169867235729, -0.3560710949285466, 0.3659175213497381)
    (0.4951951927279397, -0.5128918983128626, -0.5081034550556173, 0.48327476729569796)
    (-0.3630978950590552, 0.3607299713498126, 0.5980064525714099, -0.6167836647128428)
    
    
    
    
    
    
    
    #!!! 2024-Jun 's version calibration result:
    # F120 = {
    # 'fxn': 0.7478020191192627,
    # 'fyn': 0.39888936281204224,
    # 'fzn': 1.69412100315094,
    # 'fxc': 0.12453147768974304,
    # 'fyc': 0.023205328732728958,
    # 'fzc': 0.17563281953334808,
    # }     
    
    # FL120 = {
    # 'fxn': 0.6823626160621643,
    # 'fyn': 0.6535221934318542,
    # 'fzn': 67.79759216308594,
    # 'fxc': -0.0029870772268623114,
    # 'fyc': -0.574974536895752,
    # 'fzc': -0.31926506757736206,
    # }   
    
    # FR120 = {
    # 'fxn': 0.5174974799156189,
    # 'fyn': -0.8920024633407593,
    # 'fzn': -67.4144287109375,
    # 'fxc': 0.7106829285621643,
    # 'fyc': 0.8196682929992676,
    # 'fzc': 0.5588372945785522,
    # } 
   
    # RL = {
    # 'fxn': 0.8304807543754578,
    # 'fyn': -0.27166205644607544,
    # 'fzn': 151.30245971679688,
    # 'fxc': -0.39371949434280396,
    # 'fyc': 0.16318558156490326,
    # 'fzc': 0.4732438027858734,
    # }
   
    
    # RR = {
    # 'fxn': 1.896492600440979,
    # 'fyn': 0.7620489597320557,
    # 'fzn': -151.93609619140625,
    # 'fxc': -0.18918843567371368,
    # 'fyc': 0.7620489597320557,
    # 'fzc': 0.1966867595911026
    # }
    
    # R = {
    # 'fxn': -2.2612860202789307,
    # 'fyn': -1.897634506225586,
    # 'fzn': 178.9329376220703,
    # 'fxc': -0.28133028745651245,
    # 'fyc': 0.5165433287620544,
    # 'fzc': 0.10230113565921783
    # }
    
    # !!! 2024-Feb 's version calibration result:
    # F120 = {
    # 'fxn': 0.9732222557067871,
    # 'fyn': 0.13335508108139038,
    # 'fzn': 2.095618486404419,
    # 'fxc': -0.07370994240045547,
    # 'fyc': 0.024520529434084892,
    # 'fzc': -0.17564187943935394,
    # }     
    
    # FL120 = {
    # 'fxn': 0.7091734409332275,
    # 'fyn': 0.3381761312484741,
    # 'fzn': 67.89298248291016,
    # 'fxc': -0.12577678263187408,
    # 'fyc': 0.23768751323223114,
    # 'fzc': -0.5289285778999329,
    # }   
    
    # FR120 = {
    # 'fxn': 1.0962799787521362,
    # 'fyn': -0.33374258875846863,
    # 'fzn': -66.79678344726562,
    # 'fxc': -0.03324808180332184,
    # 'fyc': -0.014599476009607315,
    # 'fzc': -0.07454647868871689,
    # } 
   
    # RL = {
    # 'fxn': 0.4788847267627716,
    # 'fyn': 0.16447968780994415,
    # 'fzn': 151.3782958984375,
    # 'fxc': 0.24541577696800232,
    # 'fyc': 0.33335253596305847,
    # 'fzc': 0.024203738197684288,
    # }
   
    
    # RR = {
    # 'fxn': 1.3843307495117188,
    # 'fyn': 0.4358459413051605,
    # 'fzn': -151.0899658203125,
    # 'fxc': 0.0210320632904768,
    # 'fyc': 0.17808659374713898,
    # 'fzc': -0.47936850786209106
    # }
    
    # R = {
    # 'fxn': -2.7012135982513428,
    # 'fyn': -0.6712470650672913,
    # 'fzn': 179.31748962402344,
    # 'fxc': 0.2679538130760193,
    # 'fyc': 0.2423640340566635,
    # 'fzc': -0.2659626007080078
    # }
    
# #Jun2024 removed the convertion(-3.14/2, 0, -3.14/2)
# (0.8321210271056028, 0.0014494400373799101, 0.007005059982771409, 0.5545479460848564)
# (0.9998280666728236, 0.00782545400021159, 0.0015090025372912934, 0.016742827548444247)
# (0.8344808182287903, 0.006066420896604508, -0.007648033158282538, -0.5509505151413546)
# (0.24699721970111752, -0.0026486720939087644, 0.007197754327282618, 0.9689858359794022)
# (0.008354434344743488, 0.0035659910874982974, -0.02126269667242115, 0.9997326566960588)
# (0.2454811254376923, 0.008204644216095135, -0.010572567925337775, -0.9693089918465604)

# #Mar-21-2024, remove the convertion(-3.14/2, 0, -3.14/2)
# (0.8321210271056028, 0.0014494400373799101, 0.007005059982771409, 0.5545479460848564)
# (0.9998280666728236, 0.00782545400021159, 0.0015090025372912934, 0.016742827548444247)
# (0.8344808182287903, 0.006066420896604508, -0.007648033158282538, -0.5509505151413546)
# (0.24699721970111752, -0.0026486720939087644, 0.007197754327282618, 0.9689858359794022)
# (0.008354434344743488, 0.0035659910874982974, -0.02126269667242115, 0.9997326566960588)
# (0.2454811254376923, 0.008204644216095135, -0.010572567925337775, -0.9693089918465604)

#outputQuaternion-2024-Feb-latest
# FL120
# (0.6975617366053051, -0.6891072365851537, 0.14156435048306887, -0.1360087305376774)
# F120
# (0.5129526753793853, -0.5036182188418824, 0.4883843938307295, -0.49470084529364977)
# FR120
# (0.14097434541287887, -0.14255595767455692, 0.6858584396576288, -0.699572893712516)
# RL120
# (0.6102660689569468, -0.6057169867235729, -0.3560710949285466, 0.3659175213497381)
# R30
# (0.4951951927279397, -0.5128918983128626, -0.5081034550556173, 0.48327476729569796)

# Nusc_back : -89.0405962694, 0.22919685786400154, 89.86124500000001]

# #output-outdated
# F120
# (0.9998280666728236, 0.00782545400021159, 0.0015090025372912934, 0.016742827548444247)
# FL120
# (0.8321210271056028, 0.0014494400373799101, 0.007005059982771409, 0.5545479460848564)
# FR120
# (0.8344808182287903, 0.006066420896604508, -0.007648033158282538, -0.5509505151413546)
# RL
# (0.24699721970111752, -0.0026486720939087644, 0.007197754327282618, 0.9689858359794022)
# RR
# (0.2454811254376923, 0.008204644216095135, -0.010572567925337775, -0.9693089918465604)
# R
# (0.008354434344743488, 0.0035659910874982974, -0.02126269667242115, 0.9997326566960588)

# "rotation": [
# 0.6757265034669446,
# -0.6736266522251881,
# 0.21214015046209478,
# -0.21122827103904068
# ],
#F120
# "rotation": [
# 0.4998015430569128,
# -0.5030316162024876,
# 0.4997798114386805,
# -0.49737083824542755
# ],  

#R
#     "rotation": [
# 0.5037872666382278,
# -0.49740249788611096,
# -0.4941850223835201,
# 0.5045496097725578
# ],

    
   
