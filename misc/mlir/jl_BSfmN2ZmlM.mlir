module {
  func.func @main(%arg0: tensor<100x1xf32>, %arg1: tensor<100x1xf32>, %arg2: tensor<1x64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64x64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x1xf32>, %arg7: tensor<1xf32>, %arg8: tensor<f64>, %arg9: tensor<f64>, %arg10: tensor<f64>, %arg11: tensor<f64>, %arg12: tensor<1x64xf32>, %arg13: tensor<1x64xf32>, %arg14: tensor<f32>, %arg15: tensor<f32>, %arg16: tensor<64xf32>, %arg17: tensor<64xf32>, %arg18: tensor<f32>, %arg19: tensor<f32>, %arg20: tensor<64x64xf32>, %arg21: tensor<64x64xf32>, %arg22: tensor<f32>, %arg23: tensor<f32>, %arg24: tensor<64xf32>, %arg25: tensor<64xf32>, %arg26: tensor<f32>, %arg27: tensor<f32>, %arg28: tensor<64x1xf32>, %arg29: tensor<64x1xf32>, %arg30: tensor<f32>, %arg31: tensor<f32>, %arg32: tensor<1xf32>, %arg33: tensor<1xf32>, %arg34: tensor<f32>, %arg35: tensor<f32>) -> (tensor<1x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x1xf32>, tensor<1xf32>, tensor<f32>, tensor<1x64xf32>, tensor<1x64xf32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<f32>, tensor<f32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<f32>, tensor<f32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<f32>, tensor<f32>, tensor<1xf32>, tensor<1xf32>, tensor<f32>, tensor<f32>) {
    %cst = stablehlo.constant dense<0.00999999977> : tensor<1x100xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1x100xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_2 = stablehlo.constant dense<1.59576917> : tensor<64x100xf32>
    %cst_3 = stablehlo.constant dense<4.471500e-02> : tensor<64x100xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<64x100xf32>
    %cst_5 = stablehlo.constant dense<1.000000e+02> : tensor<f32>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.reshape %arg12 : (tensor<1x64xf32>) -> tensor<64x1xf32>
    %1 = stablehlo.reshape %arg13 : (tensor<1x64xf32>) -> tensor<64x1xf32>
    %2 = stablehlo.transpose %arg20, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %3 = stablehlo.transpose %arg21, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %4 = stablehlo.reshape %arg28 : (tensor<64x1xf32>) -> tensor<1x64xf32>
    %5 = stablehlo.reshape %arg29 : (tensor<64x1xf32>) -> tensor<1x64xf32>
    %6 = stablehlo.reshape %arg1 : (tensor<100x1xf32>) -> tensor<1x100xf32>
    %7 = stablehlo.dot_general %arg2, %arg0, contracting_dims = [0] x [1] : (tensor<1x64xf32>, tensor<100x1xf32>) -> tensor<64x100xf32>
    %8 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<64xf32>) -> tensor<64x100xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x100xf32>
    %10 = stablehlo.multiply %9, %9 : tensor<64x100xf32>
    %11 = stablehlo.multiply %10, %cst_3 : tensor<64x100xf32>
    %12 = stablehlo.add %11, %cst_4 : tensor<64x100xf32>
    %13 = stablehlo.multiply %cst_2, %9 : tensor<64x100xf32>
    %14 = stablehlo.multiply %13, %12 : tensor<64x100xf32>
    %15 = stablehlo.logistic %14 : tensor<64x100xf32>
    %16 = stablehlo.multiply %9, %15 : tensor<64x100xf32>
    %17 = stablehlo.dot_general %arg4, %16, contracting_dims = [0] x [0] : (tensor<64x64xf32>, tensor<64x100xf32>) -> tensor<64x100xf32>
    %18 = stablehlo.broadcast_in_dim %arg5, dims = [0] : (tensor<64xf32>) -> tensor<64x100xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x100xf32>
    %20 = stablehlo.multiply %19, %19 : tensor<64x100xf32>
    %21 = stablehlo.multiply %20, %cst_3 : tensor<64x100xf32>
    %22 = stablehlo.add %21, %cst_4 : tensor<64x100xf32>
    %23 = stablehlo.multiply %cst_2, %19 : tensor<64x100xf32>
    %24 = stablehlo.multiply %23, %22 : tensor<64x100xf32>
    %25 = stablehlo.logistic %24 : tensor<64x100xf32>
    %26 = stablehlo.multiply %19, %25 : tensor<64x100xf32>
    %27 = stablehlo.dot_general %arg6, %26, contracting_dims = [0] x [0] : (tensor<64x1xf32>, tensor<64x100xf32>) -> tensor<1x100xf32>
    %28 = stablehlo.broadcast_in_dim %arg7, dims = [0] : (tensor<1xf32>) -> tensor<1x100xf32>
    %29 = stablehlo.add %27, %28 : tensor<1x100xf32>
    %30 = stablehlo.subtract %29, %6 : tensor<1x100xf32>
    %31 = stablehlo.abs %30 : tensor<1x100xf32>
    %32 = stablehlo.multiply %31, %31 : tensor<1x100xf32>
    %33 = stablehlo.reduce(%32 init: %cst_1) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x100xf32>, tensor<f32>) -> tensor<f32>
    %34 = stablehlo.divide %33, %cst_5 : tensor<f32>
    %35 = stablehlo.multiply %cst, %31 : tensor<1x100xf32>
    %36 = stablehlo.add %35, %35 : tensor<1x100xf32>
    %37 = stablehlo.compare  GE, %30, %cst_0 : (tensor<1x100xf32>, tensor<1x100xf32>) -> tensor<1x100xi1>
    %38 = stablehlo.negate %36 : tensor<1x100xf32>
    %39 = stablehlo.select %37, %36, %38 : tensor<1x100xi1>, tensor<1x100xf32>
    %40 = stablehlo.reduce(%39 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x100xf32>, tensor<f32>) -> tensor<1xf32>
    %41 = stablehlo.dot_general %39, %26, contracting_dims = [1] x [1] : (tensor<1x100xf32>, tensor<64x100xf32>) -> tensor<1x64xf32>
    %42 = stablehlo.dot_general %arg6, %39, contracting_dims = [1] x [0] : (tensor<64x1xf32>, tensor<1x100xf32>) -> tensor<64x100xf32>
    %43 = stablehlo.multiply %42, %25 : tensor<64x100xf32>
    %44 = stablehlo.multiply %42, %19 : tensor<64x100xf32>
    %45 = stablehlo.subtract %cst_4, %25 : tensor<64x100xf32>
    %46 = stablehlo.multiply %25, %45 : tensor<64x100xf32>
    %47 = stablehlo.multiply %44, %46 : tensor<64x100xf32>
    %48 = stablehlo.multiply %47, %22 : tensor<64x100xf32>
    %49 = stablehlo.multiply %47, %23 : tensor<64x100xf32>
    %50 = stablehlo.multiply %48, %cst_2 : tensor<64x100xf32>
    %51 = stablehlo.add %43, %50 : tensor<64x100xf32>
    %52 = stablehlo.multiply %49, %cst_3 : tensor<64x100xf32>
    %53 = stablehlo.multiply %52, %19 : tensor<64x100xf32>
    %54 = stablehlo.add %51, %53 : tensor<64x100xf32>
    %55 = stablehlo.add %54, %53 : tensor<64x100xf32>
    %56 = stablehlo.reduce(%55 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<64x100xf32>, tensor<f32>) -> tensor<64xf32>
    %57 = stablehlo.dot_general %arg4, %55, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64x100xf32>) -> tensor<64x100xf32>
    %58 = stablehlo.multiply %57, %15 : tensor<64x100xf32>
    %59 = stablehlo.multiply %57, %9 : tensor<64x100xf32>
    %60 = stablehlo.subtract %cst_4, %15 : tensor<64x100xf32>
    %61 = stablehlo.multiply %15, %60 : tensor<64x100xf32>
    %62 = stablehlo.multiply %59, %61 : tensor<64x100xf32>
    %63 = stablehlo.multiply %62, %12 : tensor<64x100xf32>
    %64 = stablehlo.multiply %62, %13 : tensor<64x100xf32>
    %65 = stablehlo.multiply %63, %cst_2 : tensor<64x100xf32>
    %66 = stablehlo.add %58, %65 : tensor<64x100xf32>
    %67 = stablehlo.multiply %64, %cst_3 : tensor<64x100xf32>
    %68 = stablehlo.multiply %67, %9 : tensor<64x100xf32>
    %69 = stablehlo.add %66, %68 : tensor<64x100xf32>
    %70 = stablehlo.add %69, %68 : tensor<64x100xf32>
    %71 = stablehlo.reduce(%70 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<64x100xf32>, tensor<f32>) -> tensor<64xf32>
    %72 = stablehlo.dot_general %70, %arg0, contracting_dims = [1] x [0] : (tensor<64x100xf32>, tensor<100x1xf32>) -> tensor<64x1xf32>
    %73 = stablehlo.reshape %arg2 : (tensor<1x64xf32>) -> tensor<64x1xf32>
    %74 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %75 = stablehlo.reshape %arg6 : (tensor<64x1xf32>) -> tensor<1x64xf32>
    %76 = stablehlo.dot_general %55, %16, contracting_dims = [1] x [1] : (tensor<64x100xf32>, tensor<64x100xf32>) -> tensor<64x64xf32>
    %77 = stablehlo.convert %arg8 : (tensor<f64>) -> tensor<f32>
    %78 = stablehlo.convert %arg9 : (tensor<f64>) -> tensor<f32>
    %79 = stablehlo.convert %arg10 : (tensor<f64>) -> tensor<f32>
    %80 = stablehlo.convert %arg11 : (tensor<f64>) -> tensor<f32>
    %81 = stablehlo.broadcast_in_dim %78, dims = [] : (tensor<f32>) -> tensor<64x1xf32>
    %82 = stablehlo.multiply %81, %0 : tensor<64x1xf32>
    %83 = stablehlo.subtract %cst_6, %78 : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %83, dims = [] : (tensor<f32>) -> tensor<64x1xf32>
    %85 = stablehlo.multiply %84, %72 : tensor<64x1xf32>
    %86 = stablehlo.add %82, %85 : tensor<64x1xf32>
    %87 = stablehlo.broadcast_in_dim %79, dims = [] : (tensor<f32>) -> tensor<64x1xf32>
    %88 = stablehlo.multiply %87, %1 : tensor<64x1xf32>
    %89 = stablehlo.subtract %cst_6, %79 : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %89, dims = [] : (tensor<f32>) -> tensor<64x1xf32>
    %91 = stablehlo.abs %72 : tensor<64x1xf32>
    %92 = stablehlo.multiply %91, %91 : tensor<64x1xf32>
    %93 = stablehlo.multiply %90, %92 : tensor<64x1xf32>
    %94 = stablehlo.add %88, %93 : tensor<64x1xf32>
    %95 = stablehlo.subtract %cst_6, %arg14 : tensor<f32>
    %96 = stablehlo.broadcast_in_dim %95, dims = [] : (tensor<f32>) -> tensor<64x1xf32>
    %97 = stablehlo.divide %86, %96 : tensor<64x1xf32>
    %98 = stablehlo.subtract %cst_6, %arg15 : tensor<f32>
    %99 = stablehlo.broadcast_in_dim %98, dims = [] : (tensor<f32>) -> tensor<64x1xf32>
    %100 = stablehlo.divide %94, %99 : tensor<64x1xf32>
    %101 = stablehlo.sqrt %100 : tensor<64x1xf32>
    %102 = stablehlo.broadcast_in_dim %80, dims = [] : (tensor<f32>) -> tensor<64x1xf32>
    %103 = stablehlo.add %101, %102 : tensor<64x1xf32>
    %104 = stablehlo.divide %97, %103 : tensor<64x1xf32>
    %105 = stablehlo.broadcast_in_dim %77, dims = [] : (tensor<f32>) -> tensor<64x1xf32>
    %106 = stablehlo.multiply %104, %105 : tensor<64x1xf32>
    %107 = stablehlo.multiply %arg14, %78 : tensor<f32>
    %108 = stablehlo.multiply %arg15, %79 : tensor<f32>
    %109 = stablehlo.subtract %73, %106 : tensor<64x1xf32>
    %110 = stablehlo.broadcast_in_dim %78, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %111 = stablehlo.multiply %110, %arg16 : tensor<64xf32>
    %112 = stablehlo.broadcast_in_dim %83, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %113 = stablehlo.multiply %112, %71 : tensor<64xf32>
    %114 = stablehlo.add %111, %113 : tensor<64xf32>
    %115 = stablehlo.broadcast_in_dim %79, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %116 = stablehlo.multiply %115, %arg17 : tensor<64xf32>
    %117 = stablehlo.broadcast_in_dim %89, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %118 = stablehlo.abs %71 : tensor<64xf32>
    %119 = stablehlo.multiply %118, %118 : tensor<64xf32>
    %120 = stablehlo.multiply %117, %119 : tensor<64xf32>
    %121 = stablehlo.add %116, %120 : tensor<64xf32>
    %122 = stablehlo.subtract %cst_6, %arg18 : tensor<f32>
    %123 = stablehlo.broadcast_in_dim %122, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %124 = stablehlo.divide %114, %123 : tensor<64xf32>
    %125 = stablehlo.subtract %cst_6, %arg19 : tensor<f32>
    %126 = stablehlo.broadcast_in_dim %125, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %127 = stablehlo.divide %121, %126 : tensor<64xf32>
    %128 = stablehlo.sqrt %127 : tensor<64xf32>
    %129 = stablehlo.broadcast_in_dim %80, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %130 = stablehlo.add %128, %129 : tensor<64xf32>
    %131 = stablehlo.divide %124, %130 : tensor<64xf32>
    %132 = stablehlo.broadcast_in_dim %77, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %133 = stablehlo.multiply %131, %132 : tensor<64xf32>
    %134 = stablehlo.multiply %arg18, %78 : tensor<f32>
    %135 = stablehlo.multiply %arg19, %79 : tensor<f32>
    %136 = stablehlo.subtract %arg3, %133 : tensor<64xf32>
    %137 = stablehlo.broadcast_in_dim %78, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %138 = stablehlo.multiply %137, %2 : tensor<64x64xf32>
    %139 = stablehlo.broadcast_in_dim %83, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %140 = stablehlo.multiply %139, %76 : tensor<64x64xf32>
    %141 = stablehlo.add %138, %140 : tensor<64x64xf32>
    %142 = stablehlo.broadcast_in_dim %79, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %143 = stablehlo.multiply %142, %3 : tensor<64x64xf32>
    %144 = stablehlo.broadcast_in_dim %89, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %145 = stablehlo.abs %76 : tensor<64x64xf32>
    %146 = stablehlo.multiply %145, %145 : tensor<64x64xf32>
    %147 = stablehlo.multiply %144, %146 : tensor<64x64xf32>
    %148 = stablehlo.add %143, %147 : tensor<64x64xf32>
    %149 = stablehlo.subtract %cst_6, %arg22 : tensor<f32>
    %150 = stablehlo.broadcast_in_dim %149, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %151 = stablehlo.divide %141, %150 : tensor<64x64xf32>
    %152 = stablehlo.subtract %cst_6, %arg23 : tensor<f32>
    %153 = stablehlo.broadcast_in_dim %152, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %154 = stablehlo.divide %148, %153 : tensor<64x64xf32>
    %155 = stablehlo.sqrt %154 : tensor<64x64xf32>
    %156 = stablehlo.broadcast_in_dim %80, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %157 = stablehlo.add %155, %156 : tensor<64x64xf32>
    %158 = stablehlo.divide %151, %157 : tensor<64x64xf32>
    %159 = stablehlo.broadcast_in_dim %77, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %160 = stablehlo.multiply %158, %159 : tensor<64x64xf32>
    %161 = stablehlo.multiply %arg22, %78 : tensor<f32>
    %162 = stablehlo.multiply %arg23, %79 : tensor<f32>
    %163 = stablehlo.subtract %74, %160 : tensor<64x64xf32>
    %164 = stablehlo.multiply %110, %arg24 : tensor<64xf32>
    %165 = stablehlo.multiply %112, %56 : tensor<64xf32>
    %166 = stablehlo.add %164, %165 : tensor<64xf32>
    %167 = stablehlo.multiply %115, %arg25 : tensor<64xf32>
    %168 = stablehlo.abs %56 : tensor<64xf32>
    %169 = stablehlo.multiply %168, %168 : tensor<64xf32>
    %170 = stablehlo.multiply %117, %169 : tensor<64xf32>
    %171 = stablehlo.add %167, %170 : tensor<64xf32>
    %172 = stablehlo.subtract %cst_6, %arg26 : tensor<f32>
    %173 = stablehlo.broadcast_in_dim %172, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %174 = stablehlo.divide %166, %173 : tensor<64xf32>
    %175 = stablehlo.subtract %cst_6, %arg27 : tensor<f32>
    %176 = stablehlo.broadcast_in_dim %175, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %177 = stablehlo.divide %171, %176 : tensor<64xf32>
    %178 = stablehlo.sqrt %177 : tensor<64xf32>
    %179 = stablehlo.add %178, %129 : tensor<64xf32>
    %180 = stablehlo.divide %174, %179 : tensor<64xf32>
    %181 = stablehlo.multiply %180, %132 : tensor<64xf32>
    %182 = stablehlo.multiply %arg26, %78 : tensor<f32>
    %183 = stablehlo.multiply %arg27, %79 : tensor<f32>
    %184 = stablehlo.subtract %arg5, %181 : tensor<64xf32>
    %185 = stablehlo.broadcast_in_dim %78, dims = [] : (tensor<f32>) -> tensor<1x64xf32>
    %186 = stablehlo.multiply %185, %4 : tensor<1x64xf32>
    %187 = stablehlo.broadcast_in_dim %83, dims = [] : (tensor<f32>) -> tensor<1x64xf32>
    %188 = stablehlo.multiply %187, %41 : tensor<1x64xf32>
    %189 = stablehlo.add %186, %188 : tensor<1x64xf32>
    %190 = stablehlo.broadcast_in_dim %79, dims = [] : (tensor<f32>) -> tensor<1x64xf32>
    %191 = stablehlo.multiply %190, %5 : tensor<1x64xf32>
    %192 = stablehlo.broadcast_in_dim %89, dims = [] : (tensor<f32>) -> tensor<1x64xf32>
    %193 = stablehlo.abs %41 : tensor<1x64xf32>
    %194 = stablehlo.multiply %193, %193 : tensor<1x64xf32>
    %195 = stablehlo.multiply %192, %194 : tensor<1x64xf32>
    %196 = stablehlo.add %191, %195 : tensor<1x64xf32>
    %197 = stablehlo.subtract %cst_6, %arg30 : tensor<f32>
    %198 = stablehlo.broadcast_in_dim %197, dims = [] : (tensor<f32>) -> tensor<1x64xf32>
    %199 = stablehlo.divide %189, %198 : tensor<1x64xf32>
    %200 = stablehlo.subtract %cst_6, %arg31 : tensor<f32>
    %201 = stablehlo.broadcast_in_dim %200, dims = [] : (tensor<f32>) -> tensor<1x64xf32>
    %202 = stablehlo.divide %196, %201 : tensor<1x64xf32>
    %203 = stablehlo.sqrt %202 : tensor<1x64xf32>
    %204 = stablehlo.broadcast_in_dim %80, dims = [] : (tensor<f32>) -> tensor<1x64xf32>
    %205 = stablehlo.add %203, %204 : tensor<1x64xf32>
    %206 = stablehlo.divide %199, %205 : tensor<1x64xf32>
    %207 = stablehlo.broadcast_in_dim %77, dims = [] : (tensor<f32>) -> tensor<1x64xf32>
    %208 = stablehlo.multiply %206, %207 : tensor<1x64xf32>
    %209 = stablehlo.multiply %arg30, %78 : tensor<f32>
    %210 = stablehlo.multiply %arg31, %79 : tensor<f32>
    %211 = stablehlo.subtract %75, %208 : tensor<1x64xf32>
    %212 = stablehlo.reshape %78 : (tensor<f32>) -> tensor<1xf32>
    %213 = stablehlo.multiply %212, %arg32 : tensor<1xf32>
    %214 = stablehlo.reshape %83 : (tensor<f32>) -> tensor<1xf32>
    %215 = stablehlo.multiply %214, %40 : tensor<1xf32>
    %216 = stablehlo.add %213, %215 : tensor<1xf32>
    %217 = stablehlo.reshape %79 : (tensor<f32>) -> tensor<1xf32>
    %218 = stablehlo.multiply %217, %arg33 : tensor<1xf32>
    %219 = stablehlo.reshape %89 : (tensor<f32>) -> tensor<1xf32>
    %220 = stablehlo.abs %40 : tensor<1xf32>
    %221 = stablehlo.multiply %220, %220 : tensor<1xf32>
    %222 = stablehlo.multiply %219, %221 : tensor<1xf32>
    %223 = stablehlo.add %218, %222 : tensor<1xf32>
    %224 = stablehlo.subtract %cst_6, %arg34 : tensor<f32>
    %225 = stablehlo.reshape %224 : (tensor<f32>) -> tensor<1xf32>
    %226 = stablehlo.divide %216, %225 : tensor<1xf32>
    %227 = stablehlo.subtract %cst_6, %arg35 : tensor<f32>
    %228 = stablehlo.reshape %227 : (tensor<f32>) -> tensor<1xf32>
    %229 = stablehlo.divide %223, %228 : tensor<1xf32>
    %230 = stablehlo.sqrt %229 : tensor<1xf32>
    %231 = stablehlo.reshape %80 : (tensor<f32>) -> tensor<1xf32>
    %232 = stablehlo.add %230, %231 : tensor<1xf32>
    %233 = stablehlo.divide %226, %232 : tensor<1xf32>
    %234 = stablehlo.reshape %77 : (tensor<f32>) -> tensor<1xf32>
    %235 = stablehlo.multiply %233, %234 : tensor<1xf32>
    %236 = stablehlo.multiply %arg34, %78 : tensor<f32>
    %237 = stablehlo.multiply %arg35, %79 : tensor<f32>
    %238 = stablehlo.subtract %arg7, %235 : tensor<1xf32>
    %239 = stablehlo.reshape %109 : (tensor<64x1xf32>) -> tensor<1x64xf32>
    %240 = stablehlo.transpose %163, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %241 = stablehlo.reshape %211 : (tensor<1x64xf32>) -> tensor<64x1xf32>
    %242 = stablehlo.reshape %86 : (tensor<64x1xf32>) -> tensor<1x64xf32>
    %243 = stablehlo.reshape %94 : (tensor<64x1xf32>) -> tensor<1x64xf32>
    %244 = stablehlo.transpose %141, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %245 = stablehlo.transpose %148, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %246 = stablehlo.reshape %189 : (tensor<1x64xf32>) -> tensor<64x1xf32>
    %247 = stablehlo.reshape %196 : (tensor<1x64xf32>) -> tensor<64x1xf32>
    return %239, %136, %240, %184, %241, %238, %34, %242, %243, %107, %108, %114, %121, %134, %135, %244, %245, %161, %162, %166, %171, %182, %183, %246, %247, %209, %210, %216, %223, %236, %237 : tensor<1x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x1xf32>, tensor<1xf32>, tensor<f32>, tensor<1x64xf32>, tensor<1x64xf32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<f32>, tensor<f32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<f32>, tensor<f32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<f32>, tensor<f32>, tensor<1xf32>, tensor<1xf32>, tensor<f32>, tensor<f32>
  }
}