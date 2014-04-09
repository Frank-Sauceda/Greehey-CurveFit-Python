__author__ = "Nuno Lages"
__email__ = "lages@uthscsa.edu"


import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage as cpi
import cellprofiler.objects as cpo
import scipy as sp
cat = sp.concatenate
import scipy.stats as stats
import scipy.io as sio
from time import time
import matplotlib.pyplot as plt

from os.path import expanduser
home = expanduser("~")


class TruncThresholdObjects(cpm.CPModule):

    variable_revision_number = 1
    module_name = "TruncThresholdObjects"
    category = "Image Processing"

    def create_settings(self):

        self.input_image_name = cps.ImageNameSubscriber(
            # The text to the left of the edit box
            "Input image name:",
            # HTML help that gets displayed when the user presses the
            # help button to the right of the edit box
            doc = """This is the image that the module operates on. You can
            choose any image that is made available by a prior module.
            <br>
            <b>ImageTemplate</b> will do something to this image.
            """
        )

        self.output_image_name = cps.ImageNameProvider(
            "Output image name:",
            # The second parameter holds a suggested name for the image.
            "OutputImage",
            doc="""This is the image resulting from the operation."""
        )

        self.input_objects_name = cps.ObjectNameSubscriber(
            # The text to the left of the edit box
            "Input objects name:",
            # HTML help that gets displayed when the user presses the
            # help button to the right of the edit box
            doc = """This is the objects that the module operates on. You can
            choose any objects that is made available by a prior module.
            <br>
            <b>TruncThresholdObjects</b> will do something to this objects.
            """
        )

        self.percentile_r = cps.Float(
            "Percentile red channel:",
            # The default value
            0.99,
            doc=""""""
        )

        self.percentile_g = cps.Float(
            "Percentile green channel:",
            # The default value
            0.99,
            doc=""""""
        )

        self.percentile_b = cps.Float(
            "Percentile blue channel:",
            # The default value
            1.0,
            doc=""""""
        )

        self.percentile_k = cps.Float(
            "Percentile gray image:",
            # The default value
            1.0,
            doc=""""""
        )
        self.npoints = cps.Integer(
            "Number of points in Gaussian kernel density:",
            50,
            doc=""""""
        )

    def settings(self):
        return [self.input_image_name,
                self.output_image_name,
                self.input_objects_name,
                self.percentile_r,
                self.percentile_g,
                self.percentile_b,
                self.percentile_k,
                self.npoints]

    def run(self, workspace):

        t0 = time()

        diagnostics = dict()

        npoints = self.npoints.get_value()

        input_objects_name = self.input_objects_name.value
        object_set = workspace.object_set
        assert isinstance(object_set, cpo.ObjectSet)

        input_image_name = self.input_image_name.value
        image_set = workspace.image_set
        assert isinstance(image_set, cpi.ImageSet)
        output_image_name = self.output_image_name.value

        input_image = image_set.get_image(input_image_name)# must_be_rgb=True)
        pixels = input_image.pixel_data
        diagnostics['pixels'] = pixels

        input_objects = object_set.get_objects(input_objects_name)

        mask = input_objects.get_segmented()

        new_im = sp.zeros(shape=pixels.shape)

        diagnostics['new_im'] = list()
        diagnostics['nucleus_processed'] = list()
        diagnostics['nucleus_pixels'] = list()
        diagnostics['ci'] = list()

        diagnostics['time_first_part'] = time() - t0

        for x in range(1, mask.max()+1):

            t0 = time()

            nucleus_map = mask == x

            if len(pixels.shape) == 3:  # rgb
                nucleus_pixels = \
                    sp.multiply(pixels, nucleus_map[:, :, sp.newaxis] > 0)
            elif len(pixels.shape) == 2:  # grey scale
                nucleus_pixels = \
                    sp.multiply(pixels, nucleus_map > 0)

            diagnostics['times_loop_' + str(x) + '_nditer'] = time() - t0
            t0 = time()

            diagnostics['nucleus_pixels'].append(nucleus_pixels)

            # sio.savemat(home + '/diagnostics0.mat', diagnostics)

            if len(nucleus_pixels.shape) == 3:

                nucleus_pixels_t = sp.transpose(nucleus_pixels)

                nucleus_r = \
                    nucleus_pixels_t[0][sp.nonzero(nucleus_pixels_t[0])]
                diagnostics['nucleus_r'] = nucleus_r
                nucleus_ci_r = var_ksFit(nucleus_r,
                                         npoints,
                                         self.percentile_r.get_value(),
                                         extra='red')

                nucleus_g = \
                    nucleus_pixels_t[1][sp.nonzero(nucleus_pixels_t[1])]
                diagnostics['nucleus_g'] = nucleus_g
                nucleus_ci_g = var_ksFit(nucleus_g,
                                         npoints,
                                         self.percentile_g.get_value(),
                                         extra='green')

                nucleus_b = \
                    nucleus_pixels_t[2][sp.nonzero(nucleus_pixels_t[2])]
                diagnostics['nucleus_b'] = nucleus_b
                nucleus_ci_b = var_ksFit(nucleus_b,
                                         npoints,
                                         self.percentile_b.get_value(),
                                         extra='blue')

                diagnostics['times_loop_' + str(x) + '_ci'] = time() - t0
                t0 = time()

                diagnostics['ci'].append((nucleus_ci_r, nucleus_ci_g,
                                          nucleus_ci_b))
                sio.savemat(home + '/diagnostics.mat', diagnostics)
                # diagnostics['mu'].append((mu_r, mu_g, mu_b))
                # diagnostics['sigma'].append((sigma_r, sigma_g, sigma_b))
                # diagnostics['sigma2'].append((sigma2_r, sigma2_g, sigma2_b))
                # diagnostics['old_sigma'].append(
                #     (old_sigma_r, old_sigma_g, old_sigma_b))
                # diagnostics['a'].append((a_r, a_g, a_b))
                # diagnostics['b'].append((b_r, b_g, b_b))
                # diagnostics['x1'].append((x1_r, x1_g, x1_b))
                # diagnostics['x2'].append((x2_r, x2_g, x2_b))
                # diagnostics['cx'].append((cx_r, cx_g, cx_b))
                # diagnostics['yhat'].append((yhat_r, yhat_g, yhat_b))

                nucleus_processed = update_image(nucleus_pixels,
                                                 nucleus_ci_r,
                                                 nucleus_ci_g,
                                                 nucleus_ci_b)

            elif len(nucleus_pixels.shape) == 2:

                flattened = sp.concatenate(nucleus_pixels)
                flattened = flattened[sp.nonzero(flattened)]

                nucleus_ci = var_ksFit(flattened,
                                       npoints,
                                       self.percentile_k.get_value(),
                                       extra='grey')

                nucleus_processed = sp.multiply(
                    nucleus_pixels, nucleus_pixels > nucleus_ci)


            diagnostics['times_loop_' + str(x) + '_update'] = time() - t0

            diagnostics['nucleus_processed'].append(nucleus_processed)

            new_im = new_im + nucleus_processed

            diagnostics['new_im'].append(new_im)

            sio.savemat(home + '/diagnostics.mat', diagnostics)

        output_image = cpi.Image(new_im, parent_image=input_image)
        image_set.add(output_image_name, output_image)

    def is_interactive(self):
        return False


def var_ksFit(data, npoints, perc, extra=None):

    diag_vksf = dict()
    diag_vksf['data'] = data
    diag_vksf['npoints'] = npoints
    diag_vksf['perc'] = perc

    sio.savemat(home + '/diag_vksf.mat', diag_vksf)

    # kde_pdf = stats.gaussian_kde(flattened)
    kde_pdf = stats.gaussian_kde(data)

    # xi, dx = sp.linspace(flattened.min(), flattened.max(), npoints, retstep=True)
    xi, dx = sp.linspace(data.min(), data.max(), npoints, retstep=True)
    diag_vksf['xi'] = xi
    diag_vksf['dx'] = dx

    f = kde_pdf(xi)
    diag_vksf['f'] = f

    plt.figure()
    plt.title(extra)
    # plt.hist(flattened, bins=npoints, color=extra)
    plt.hist(data, bins=npoints, color=extra, alpha=0.5)

    mdx = sp.where(f == f.max())#[0][0]
    diag_vksf['mdx'] = mdx
    mu = xi[mdx]
    diag_vksf['mu'] = mu
    # sigma = sp.std(flattened)
    sigma = sp.std(data)
    diag_vksf['sigma'] = sigma

    err_lookforward = sp.int_(sp.floor(mdx + 0.5 * sigma / dx))
    diag_vksf['err_lookforward'] = err_lookforward

    diag_vksf['sigma_hat_0'] = list()
    diag_vksf['sigma_hat_1'] = list()
    diag_vksf['mu_hat_0'] = list()
    diag_vksf['mu_hat_1'] = list()
    diag_vksf['local_norm'] = list()
    diag_vksf['y_sigma'] = list()
    diag_vksf['y_mu'] = list()
    diag_vksf['s_sigma'] = list()
    diag_vksf['s_mu'] = list()
    diag_vksf['my_sigma'] = list()
    diag_vksf['my_mu'] = list()
    diag_vksf['delta_sigma'] = list()
    diag_vksf['delta_mu'] = list()
    diag_vksf['ci'] = list()

    for kk in xrange(3):

        sigma_hat = sp.arange(sigma*0.5, sigma*1.5 + sigma/200, sigma/200)
        diag_vksf['sigma_hat_0'].append(sigma_hat)

        delta = list()
        for i in xrange(len(sigma_hat)):
            local_norm = stats.norm(mu, sigma_hat[i])
            y = local_norm.pdf(xi)
            my = y.max()
            s = (y[sp.arange(0, err_lookforward)]/my
                 - f[sp.arange(0, err_lookforward)]/f.max()) ** 2
            delta.append(s.sum())
        diag_vksf['y_sigma'].append(y)
        diag_vksf['my_sigma'].append(my)
        diag_vksf['s_sigma'].append(s)
        diag_vksf['delta_sigma'].append(delta)
        delta = sp.array(delta)

        mx, mdx = delta.min(), sp.where(delta == delta.min())
        diag_vksf['mx_sigma'], diag_vksf['mdx_sigma'] = mx, mdx
        sigma_hat = sigma_hat[mdx]
        sigma = sigma_hat
        diag_vksf['sigma_hat_1'].append(sigma_hat)

        mu_hat = sp.arange(mu * 0.5, mu * 1.5 + mu/200, mu/200)
        diag_vksf['mu_hat_0'].append(mu_hat)

        delta = list()
        for i in xrange(len(mu_hat)):
            local_norm = stats.norm(mu_hat[i], sigma_hat)
            y = local_norm.pdf(xi)
            my = y.max()
            s = (y[sp.arange(0, err_lookforward)]/my
                 - f[sp.arange(0, err_lookforward)]/f.max()) ** 2
            delta.append(s.sum())
        diag_vksf['y_mu'].append(y)
        diag_vksf['my_mu'].append(my)
        diag_vksf['s_mu'].append(s)
        diag_vksf['delta_mu'].append(delta)
        delta = sp.array(delta)

        sio.savemat(home + '/diag_vksf.mat', diag_vksf)

        mx, mdx = delta.min(), sp.where(delta == delta.min())
        diag_vksf['mx_mu'], diag_vksf['mdx_mu'] = mx, mdx
        mu_hat = mu_hat[mdx]
        mu = mu_hat
        diag_vksf['mu_hat_1'].append(mu_hat)

        local_norm = stats.norm(mu_hat, sigma_hat)
        y = local_norm.pdf(xi)

        ci = local_norm.ppf(perc)
        diag_vksf['ci'].append(ci)
        sio.savemat(home + '/diag_vksf.mat', diag_vksf)
        # plt.plot(xi, y * f.max()/y.max() * len(flattened) * dx,
        plt.plot(xi, y * f.max()/y.max() * len(data) * dx,
                 marker='', linestyle='--', color='k')
        plt.plot((ci, ci), plt.ylim(), marker='',
                 linestyle='-', color='k')
        plt.savefig(home + '/cell_profiler_hist_' + extra + str(kk) + '.pdf')

    return ci


def update_image(original_im, ci_red, ci_green, ci_blue):

    ci_vec = sp.transpose(sp.array((ci_red, ci_green, ci_blue)))
    ci_matrix = sp.multiply(sp.ones(original_im.shape),
                            # sp.array(sp.newaxis, ci_vec))
                            ci_vec)
    new_im = sp.multiply(original_im, original_im > ci_matrix)

    return new_im
